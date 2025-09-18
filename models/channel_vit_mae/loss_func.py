import os

## add the parent directory to the path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import get_rank
import torch.distributed.nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn


def pairwise_distance_v2(proxies, x, squared=False):
    if squared:
        return (torch.cdist(x, proxies, p=2)) ** 2
    else:
        return torch.cdist(x, proxies, p=2)


def compute_proxy_loss(proxies, img_emb, gt_imgs, scale: float | nn.Parameter) -> Tensor:
    """
    proxies: shape of (num_classes, dim)
    img_emb: shape of (num_imgs, dim)
    gt_imgs: shape of (num_imgs)
    """
    proxies_emb = scale * F.normalize(proxies, p=2, dim=-1)
    img_emb = scale * F.normalize(img_emb, p=2, dim=-1)

    img_dist = pairwise_distance_v2(proxies=proxies_emb, x=img_emb, squared=True)
    img_dist = img_dist * -1.0

    cross_entropy = nn.CrossEntropyLoss(reduction="mean")
    img_loss = cross_entropy(img_dist, gt_imgs)
    return img_loss


class FourierLoss(nn.Module):
    def __init__(
        self,
        use_l1_loss: bool = True,
        # num_multimodal_modalities: int = 1,  # set to 1 for vanilla MAE, 6 for channel-agnostic MAE
    ) -> None:
        """
        Recursion Pharmaceuticals 2024
        Fourier transform loss is only sound when using L1 or L2 loss to compare the frequency domains
        between the images / their radial histograms.

        We will always set `reduction="none"` and enforce that the computation of any reductions from the
        output of this loss be managed by the model under question.
        """
        super().__init__()
        self.loss = nn.L1Loss(reduction="none") if use_l1_loss else nn.MSELoss(reduction="none")
        # self.num_modalities = num_multimodal_modalities

    def forward(self, input: torch.Tensor, target: torch.Tensor, num_channels: int) -> torch.Tensor:
        # input = reconstructed image, target = original image
        # flattened images from MAE are (B, H*W, C), so, here we convert to B x C x H x W (note we assume H == W)
        flattened_images = len(input.shape) == len(target.shape) == 3
        if flattened_images:
            B, H_W, C = input.shape
            H_W = H_W // num_channels  ## self.num_modalities
            four_d_shape = (B, -1, int(H_W**0.5), int(H_W**0.5))  ## (B, C, H, W)

            input = input.view(*four_d_shape)
            target = target.view(*four_d_shape)
        else:
            B, C, h, w = input.shape
            H_W = h * w

        if len(input.shape) != len(target.shape) != 4:
            raise ValueError(f"Invalid input shape: got {input.shape} and {target.shape}.")

        fft_reconstructed = torch.fft.fft2(input)
        fft_original = torch.fft.fft2(target)

        magnitude_reconstructed = torch.abs(fft_reconstructed)
        magnitude_original = torch.abs(fft_original)

        loss_tensor: torch.Tensor = self.loss(magnitude_reconstructed, magnitude_original)

        # if (
        #     flattened_images and not self.num_bins
        # ):  # then output loss should be reshaped
        if flattened_images:
            loss_tensor = loss_tensor.reshape(B, -1, C)

        return loss_tensor


def compute_cross_entropy(p, q):
    q = F.log_softmax(q, dim=-1)
    loss = torch.sum(p * q, dim=-1)
    return -loss.mean()


def stablize_logits(logits):
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()
    return logits


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class MultiPosConLoss(nn.Module):
    """
    Multi-Positive Contrastive Loss: https://arxiv.org/pdf/2306.00984.pdf
    Code adapted from https://github.com/google-research/syn-rep-learn/blob/main/StableRep/models/losses.py#L49
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def set_temperature(self, temp=0.1):
        self.temperature = temp

    def forward(self, feats, labels):
        """
        Args:
            feats: shape [B, D]
            labels: shape [B]
        """
        device = feats.device

        feats = F.normalize(feats, dim=-1, p=2)
        local_batch_size = feats.size(0)

        all_feats = torch.cat(torch.distributed.nn.all_gather(feats), dim=0)
        all_labels = concat_all_gather(labels)  # no gradient gather

        # compute the mask based on labels
        mask = torch.eq(labels.view(-1, 1), all_labels.contiguous().view(1, -1)).float().to(device)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(mask.shape[0]).view(-1, 1).to(device) + local_batch_size * get_rank(), 0)

        mask = mask * logits_mask

        # compute logits
        logits = torch.matmul(feats, all_feats.T) / self.temperature
        logits = logits - (1 - logits_mask) * 1e9

        # optional: minus the largest logit to stablize logits
        logits = stablize_logits(logits)

        # compute ground-truth distribution
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
        loss = compute_cross_entropy(p, logits)

        return loss


class SimCLRContrastiveLoss(nn.Module):
    """
    Distributed SimCLR Loss (local positives only, vectorized)
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def set_temperature(self, temp=0.1):
        self.temperature = temp

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: shape [2*B, D]
                First B are one set of views, next B are the other set.
                Positives are *within the local batch*, not across processes.
        """
        device = feats.device
        feats = F.normalize(feats, dim=-1, p=2)

        local_batch = feats.shape[0] // 2
        assert feats.shape[0] % 2 == 0, "feats should have even number of samples (pairs)."

        # --------- Gather across all processes ---------
        all_feats = torch.cat(torch.distributed.nn.all_gather(feats), dim=0)

        # --------- Similarity matrix ---------
        sim_matrix = torch.matmul(all_feats, all_feats.T) / self.temperature

        # mask self-similarity
        mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)

        # stabilize
        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True).values.detach()

        # --------- Restrict to local samples ---------
        rank = torch.distributed.get_rank()
        start = rank * feats.shape[0]
        end = start + feats.shape[0]
        sim_matrix_local = sim_matrix[start:end]  # [2*B_local, 2*B_global]

        # --------- Vectorized positive mask ---------
        idx = torch.arange(local_batch, device=device)
        pos_idx = torch.cat([start + local_batch + idx, start + idx])  # positives for first view  # positives for second view
        pos_mask = torch.zeros_like(sim_matrix_local, dtype=torch.bool, device=device)
        pos_mask[torch.arange(2 * local_batch, device=device), pos_idx] = True

        # --------- Compute log-softmax ---------
        log_prob = F.log_softmax(sim_matrix_local, dim=1)

        # NLL over positives
        loss = -(log_prob * pos_mask.float()).sum(dim=1)
        loss = loss.mean()
        return loss


def setup_ddp():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def main():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    batch_size = 1  # number of pairs per GPU
    feature_dim = 4

    # create synthetic features for 2B_local samples
    feats = torch.randn(batch_size * 2, feature_dim, device=device)

    loss_fn = SimCLRContrastiveLoss(temperature=0.1)
    loss = loss_fn(feats)


if __name__ == "__main__":
    torch.set_printoptions(precision=2, sci_mode=False)
    main()
