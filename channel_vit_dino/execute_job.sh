. /etc/profile.d/pixi.sh
cd CHAMMI-75/channel_vit_dino
python -m torch.distributed.launch --nproc_per_node=8 main_dino.py --config ../../$CONFIG_NAME