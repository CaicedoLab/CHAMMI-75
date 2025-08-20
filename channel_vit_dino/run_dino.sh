rm -r /mnt/cephfs/mir/jcaicedo/projects/channel_vit_dinov1/models/wandb_test
python -m torch.distributed.launch --nproc_per_node=7 main_dino.py -c ./testconfig.yaml > log.txt
