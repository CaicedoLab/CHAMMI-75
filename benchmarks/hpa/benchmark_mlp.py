import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import average_precision_score
import pandas as pd
import itertools
import os

class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha > 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean() if self.reduction == "mean" else loss.sum() if self.reduction == "sum" else loss

def build_mlp(input_dim, output_dim, hidden_sizes, dropout, activation_fn):
    layers = []
    last_dim = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last_dim, h))
        layers.append(activation_fn())
        layers.append(nn.Dropout(dropout))
        last_dim = h
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)

def train_one_model(train_x, train_y, val_x, val_y, config, device):
    model = build_mlp(
        input_dim=train_x.shape[1],
        output_dim=train_y.shape[1],
        hidden_sizes=config['hidden_sizes'],
        dropout=config['dropout'],
        activation_fn=config['activation_fn']
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = SigmoidFocalLoss()

    train_loader = DataLoader(
        TensorDataset(train_x.float().to(device), train_y.float().to(device)),
        batch_size=8192, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_x.float().to(device), val_y.float().to(device)),
        batch_size=8192
    )

    best_map = 0.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(50):  # increased to help larger models
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        y_preds, y_trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                y_preds.append(torch.sigmoid(out).cpu())
                y_trues.append(yb.cpu())

        y_pred = torch.cat(y_preds).numpy()
        y_true = torch.cat(y_trues).numpy()
        val_map = average_precision_score(y_true, y_pred, average="macro")

        if val_map > best_map:
            best_map = val_map
            best_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 6:
                break  # early stopping

    return best_map, best_state, config

def benchmark_models(
    train_x,
    train_y,
    val_x,
    val_y,
    test_x,
    test_y,
    unique_cats,
    device,
    save_folder,
    log_filename="mlp_benchmark_results.csv",
):
    import itertools
    import os
    import pandas as pd

    activations = {
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "gelu": nn.GELU,
    }

    hidden_sizes_list = [
        [256],
        [512],
        [1024],
        [512, 256],
        [1024, 512],
        [1024, 512, 256],
    ]
    dropouts = [0.0, 0.3, 0.5, 0.7]
    lrs = [1e-4, 1e-3, 3e-3]

    combinations = list(itertools.product(hidden_sizes_list, dropouts, lrs, activations.items()))
    results = []

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for idx, (hidden_sizes, dropout, lr, (act_name, act_fn)) in enumerate(combinations):
        config = {
            "hidden_sizes": hidden_sizes,
            "dropout": dropout,
            "activation": act_name,
            "activation_fn": act_fn,
            "lr": lr,
        }

        print(f"\n🔍 Trying Config {idx + 1}/{len(combinations)}: {config}")
        val_map, _, _ = train_one_model(train_x, train_y, val_x, val_y, config, device)

        results.append({
            "model_index": idx,
            "val_map": val_map,
            "hidden_sizes": str(hidden_sizes),
            "dropout": dropout,
            "activation": act_name,
            "lr": lr,
        })

    df_results = pd.DataFrame(results).sort_values(by="val_map", ascending=False)
    print("\n📊 Top 10 Models by Val MAP:")
    print(df_results.head(10))

    # Save results to CSV
    csv_path = os.path.join(save_folder, log_filename)
    df_results.to_csv(csv_path, index=False)
    print(f"\n📝 Saved benchmark results to: {csv_path}")

    return df_results

