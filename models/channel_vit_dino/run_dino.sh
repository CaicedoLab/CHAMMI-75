rm -r /mnt/cephfs/mir/jcaicedo/projects/channel_vit_dinov1/models/75ds_test /mnt/cephfs/mir/jcaicedo/projects/channel_vit_dinov1/models/chammi_test
torchrun --nproc_per_node=8 main_dino.py -c ./testconfig.yaml
