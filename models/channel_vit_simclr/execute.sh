
#export CUDA_LAUNCH_BLOCKING=1
#export TORCH_USE_CUDA_DSA=1
python -m torch.distributed.launch  --nproc_per_node=7 --nnodes=2 --node_rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=144.92.142.147:29701 main_simclr.py --data_path /scr/data/CHAMMI-75_small.zip --output_dir /scr/vidit/Models/SimCLR_CHANViT_10ds --batch_size_per_gpu 64 --num_workers 4 --epochs 100 --metadata_path /mnt/cephfs/mir/jcaicedo/morphem/dataset/sampling/CHAMMI-75_small_metadata.csv --dataset_filter 10ds --lr 1e-5 --guided_cropping True --guided_crops_path /scr/data/CHAMMI-75_guidance


#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch -nproc_per_node=7 --nnodes=2 --node_rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=144.92.142.147:29701 main_simclr.py --data_path /scr/data/chammi_train.zip --output_dir /scr/vidit/Models/SimCLR_CHANViT_CHAMMI --batch_size_per_gpu 64 --num_workers 4 --epochs 100 --metadata_path /mnt/cephfs/mir/jcaicedo/morphem/dataset/sampling/multi_channel_chammi_metadata.csv --dataset_filter none

#python -m torch.distributed.launch --nproc_per_node=8 main_simclr.py --data_path /scr/data/chammi_train.zip --output_dir /scr/vidit/Models/SimCLR_CHANViT_Allen --batch_size_per_gpu 64 --num_workers 4 --epochs 100 --metadata_path /mnt/cephfs/mir/jcaicedo/morphem/dataset/sampling/multi_channel_chammi_metadata.csv --dataset_filter allen --overwrite

#python -m torch.distributed.launch --nproc_per_node=8 main_simclr.py --data_path /scr/data/chammi_train.zip --output_dir /scr/vidit/Models/SimCLR_CHANViT_CP --batch_size_per_gpu 64 --num_workers 4 --epochs 100 --metadata_path /mnt/cephfs/mir/jcaicedo/morphem/dataset/sampling/multi_channel_chammi_metadata.csv --dataset_filter cp --overwrite

#python -m torch.distributed.launch --nproc_per_node=8 main_simclr.py --data_path /scr/data/chammi_train.zip --output_dir /scr/vidit/Models/SimCLR_CHANViT_HPA --batch_size_per_gpu 64 --num_workers 4 --epochs 100 --metadata_path /mnt/cephfs/mir/jcaicedo/morphem/dataset/sampling/multi_channel_chammi_metadata.csv --dataset_filter hpa --overwrite

#python -m torch.distributed.launch --nproc_per_node=8 main_simclr.py --data_path /scr/data/CHAMMI-75_small.zip --output_dir /scr/vidit/Models/SimCLR_CHANViT_10ds --batch_size_per_gpu 32 --num_workers 4 --epochs 100 --metadata_path /scr/data/CHAMMI-75_small_metadata.csv --dataset_filter 10ds --guided_crops_path /scr/data/CHAMMI-75_guidance --guided_cropping True
