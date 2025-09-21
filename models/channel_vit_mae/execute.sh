#python -m torch.distributed.launch --nproc_per_node=8 main_mae.py --data_path /scr/data/chammi_train.zip --output_dir /scr/vidit/Models/MAE_CHANViT_CHAMMI --batch_size_per_gpu 64 --num_workers 4 --epochs 100 --metadata_path /mnt/cephfs/mir/jcaicedo/morphem/dataset/sampling/multi_channel_chammi_metadata.csv --dataset_filter none

#python -m torch.distributed.launch --nproc_per_node=8 main_mae.py --data_path /scr/data/chammi_train.zip --output_dir /scr/vidit/Models/MAE_CHANViT_Allen --batch_size_per_gpu 64 --num_workers 4 --epochs 100 --metadata_path /mnt/cephfs/mir/jcaicedo/morphem/dataset/sampling/multi_channel_chammi_metadata.csv --dataset_filter allen

#python -m torch.distributed.launch --nproc_per_node=8 main_mae.py --data_path /scr/data/chammi_train.zip --output_dir /scr/vidit/Models/MAE_CHANViT_HPA --batch_size_per_gpu 64 --num_workers 4 --epochs 100 --metadata_path /mnt/cephfs/mir/jcaicedo/morphem/dataset/sampling/multi_channel_chammi_metadata.csv --dataset_filter hpa

#python -m torch.distributed.launch --nproc_per_node=8 main_mae.py --data_path /scr/data/chammi_train.zip --output_dir /scr/vidit/Models/MAE_CHANViT_CP --batch_size_per_gpu 64 --num_workers 4 --epochs 100 --metadata_path /mnt/cephfs/mir/jcaicedo/morphem/dataset/sampling/multi_channel_chammi_metadata.csv --dataset_filter cp

python -m torch.distributed.launch --nproc_per_node=8 main_mae.py --data_path /scr/data/CHAMMI-75_small.zip --output_dir /scr/vidit/Models/MAE_CHANViT_10ds_lr_change_warmup_change --batch_size_per_gpu 64 --num_workers 4 --epochs 100 --metadata_path /scr/data/CHAMMI-75_small_metadata.csv --dataset_filter 10ds --guided_crops_path /scr/data/CHAMMI-75_guidance --guided_cropping True


