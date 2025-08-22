rm -r /mnt/cephfs/mir/jcaicedo/projects/channel_vit_dinov1/models/75ds_test
python -m torch.distributed.launch --nproc_per_node=1 main_dino.py -c ./testconfig.yaml
