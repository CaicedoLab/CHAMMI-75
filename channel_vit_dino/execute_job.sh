. /etc/profile.d/pixi.sh
cd CHAMMI-75/channel_vit_dino
python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch channelvit_small --data_path ../../chammi_train.zip --output_dir ./chammi_vit_testing --lr 0.00005 --batch_size_per_gpu 32 --dataset_size large --metadata ../../multi_channel_chammi_metadata.csv