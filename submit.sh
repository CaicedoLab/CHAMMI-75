#!/bin/bash
#SBATCH --job-name=DINOV1   # Job name
#SBATCH --output=lumi_logs/DINOV1.%jo # Name of stdout output file
#SBATCH --error=lumi_logs/DINOV1.%je  # Name of stderr error file
#SBATCH --account=project_462000892
#SBATCH --partition=dev-g	 #standard-g #standard-g # small-g dev-g or small-g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=00-03:00:00

export RDZV_HOST=$(hostname)
export RDZV_PORT=29400
export PYTHONPATH="/pfs/lustrep1/scratch/project_462000892/sunny/code/ICLR_SUB_MAIN/CHAMMI-75:$PYTHONPATH"
export NCCL_TIMEOUT=1800  # Increase timeout from 10min to 30min
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

module use /appl/local/csc/modulefiles/
module load pytorch/2.4
pip install wandb==0.19.0
pip install albumentations
pip install munkres
#srun python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 main_ibot.py --arch vit_base --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --norm_last_layer false --epochs 300 --batch_size_per_gpu 64 --shared_head true --out_dim 8192 --local_crops_number 10 --global_crops_scale 0.25 1 --local_crops_scale 0.05 0.25 --pred_ratio 0 0.3 --pred_ratio_var 0 0.2 --pred_start_epoch 50 --data_path SingleCellDataset --momentum_teacher 0.9995 --num_workers 12 --in_chans 1 --output_dir one_channel_jitter_500k --load_from /scratch/project_462000555/sunny/code/ibot/benchmark-self-supervised/Ibot/one_channel_jitter_500k/checkpoint0110.pth
#srun python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 main_ibot.py --arch vit_base --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --norm_last_layer false --epochs 300 --batch_size_per_gpu 64 --shared_head true --out_dim 8192 --local_crops_number 10 --global_crops_scale 0.25 1 --local_crops_scale 0.05 0.25 --pred_ratio 0 0.3 --pred_ratio_var 0 0.2 --pred_start_epoch 50 --data_path SingleCellDataset --momentum_teacher 0.9995 --num_workers 12 --in_chans 1 --output_dir one_channel_jitter_500k_channel
#srun python3 -m torch.distributed.run --standalone --nnodes=2 --nproc_per_node=8 main_ibot.py --arch vit_base --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --norm_last_layer false --epochs 300 --batch_size_per_gpu 64 --shared_head true --out_dim 8192 --local_crops_number 10 --global_crops_scale 0.25 1 --local_crops_scale 0.05 0.25 --pred_ratio 0 0.3 --pred_ratio_var 0 0.2 --pred_start_epoch 50 --data_path SingleCellDataset --momentum_teacher 0.9995 --num_workers 7 --in_chans 1 --output_dir test_large

#srun python3 -m torch.distributed.run \
#    --nnodes=$SLURM_JOB_NUM_NODES \
#    --nproc_per_node=8 \
#    --rdzv_id=$SLURM_JOB_ID \
#    --rdzv_backend=c10d \
#    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
#    main_ibot.py --arch vit_base --lr 0.000125 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --norm_last_layer false --epochs 300 --batch_size_per_gpu 64 --shared_head true --out_dim 8192 --local_crops_number 10 --global_crops_scale 0.25 1 --local_crops_scale 0.05 0.25 --pred_ratio 0 0.3 --pred_ratio_var 0 0.2 --pred_start_epoch 50 --data_path SingleCellDataset --momentum_teacher 0.9995 --num_workers 7 --in_chans 1 --output_dir test_large_new_lr


#srun python3 -m torch.distributed.run \
#    --nnodes=$SLURM_JOB_NUM_NODES \
#    --nproc_per_node=8 \
#    --rdzv_id=$SLURM_JOB_ID \
#    --rdzv_backend=c10d \
#    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
#    main_ibot.py --arch vit_base --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --norm_last_layer false --epochs 300 --batch_size_per_gpu 64 --shared_head true --out_dim 8192 --local_crops_number 10 --global_crops_scale 0.25 1 --local_crops_scale 0.05 0.25 --pred_ratio 0 0.3 --pred_ratio_var 0 0.2 --pred_start_epoch 50 --data_path SingleCellDataset --momentum_teacher 0.9995 --num_workers 7 --in_chans 1 --output_dir test_large --load_from /scratch/project_462000555/sunny/code/ibot/benchmark-self-supervised/Ibot/test_large/checkpoint.pth

#srun python3 -m torch.distributed.run \
#    --nnodes=$SLURM_JOB_NUM_NODES \
#    --nproc_per_node=8 \
#    --rdzv_id=$SLURM_JOB_ID \
#    --rdzv_backend=c10d \
#    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
#    main_ibot.py --arch vit_base --lr 0.000125 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --norm_last_layer false --epochs 300 --batch_size_per_gpu 64 --shared_head true --out_dim 8192 --local_crops_number 10 --global_crops_scale 0.25 1 --local_crops_scale 0.05 0.25 --pred_ratio 0 0.3 --pred_ratio_var 0 0.2 --pred_start_epoch 50 --data_path SingleCellDataset --momentum_teacher 0.9995 --num_workers 7 --in_chans 1 --output_dir test_large_new_lr --load_from /scratch/project_462000555/sunny/code/ibot/benchmark-self-supervised/Ibot/test_large_new_lr/checkpoint.pth

#srun python3 -m torch.distributed.run \
#    --nnodes=$SLURM_JOB_NUM_NODES \
#    --nproc_per_node=8 \
#    --rdzv_id=$SLURM_JOB_ID \
#    --rdzv_backend=c10d \
#    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
#    main_ibot.py --arch vit_base --lr 0.000125 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --norm_last_layer false --epochs 300 --batch_size_per_gpu 32 --shared_head true --out_dim 8192 --local_crops_number 10 --global_crops_scale 0.25 1 --local_crops_scale 0.05 0.09 --pred_ratio 0 0.3 --pred_ratio_var 0 0.2 --pred_start_epoch 50 --data_path SingleCellDataset --momentum_teacher 0.9995 --num_workers 7 --in_chans 1 --output_dir large_scale_resize --load_from /scratch/project_462000555/sunny/code/ibot/benchmark-self-supervised/Ibot/large_scale_resize/checkpoint.pth

#srun python3 -m torch.distributed.run \
#    --nnodes=$SLURM_JOB_NUM_NODES \
#    --nproc_per_node=8 \
#    --rdzv_id=$SLURM_JOB_ID \
#    --rdzv_backend=c10d \
#    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
#    main_ibot.py --arch vit_base --lr 0.000125 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --norm_last_layer false --epochs 300 --batch_size_per_gpu 32 --shared_head true --out_dim 8192 --local_crops_number 10 --global_crops_scale 0.2 0.4 --local_crops_scale 0.05 0.09 --pred_ratio 0 0.3 --pred_ratio_var 0 0.2 --pred_start_epoch 50 --data_path SingleCellDataset --momentum_teacher 0.9995 --num_workers 7 --in_chans 1 --output_dir chammiv2_2_resize_both  --load_from /scratch/project_462000555/sunny/code/ibot/benchmark-self-supervised/Ibot/chammiv2_2_resize_both/checkpoint.pth

#srun python3 -m torch.distributed.run \
#    --nnodes=$SLURM_JOB_NUM_NODES \
#    --nproc_per_node=8 \
#    --rdzv_id=$SLURM_JOB_ID \
#    --rdzv_backend=c10d \
#    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
#    main_ibot.py --arch vit_base --lr 0.000125 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --norm_last_layer false --epochs 300 --batch_size_per_gpu 32 --shared_head true --out_dim 8192 --local_crops_number 10 --global_crops_scale 0.25 1 --local_crops_scale 0.05 0.25 --pred_ratio 0 0.3 --pred_ratio_var 0 0.2 --pred_start_epoch 50 --data_path SingleCellDataset --momentum_teacher 0.9995 --num_workers 7 --in_chans 1 --output_dir chammiv2_2_original_size --load_from /scratch/project_462000555/sunny/code/ibot/benchmark-self-supervised/Ibot/chammiv2_2_original_size/checkpoint.pth

srun python3 -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
    dinov1/main_dino.py --arch vit_small \
    --data_path /scratch/project_462000892/sunny/data/CHAMMI-75_train.zip \
    --output_dir dino_result_1 --lr 0.00005 --batch_size_per_gpu 320 \
    --guided_crops_path /scratch/project_462000892/sunny/data/CHAMMI-75_guidance.zip \
    --multiscale True --dataset_size large --guided_cropping True --num_workers 3
