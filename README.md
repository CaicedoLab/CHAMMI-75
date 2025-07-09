# CHAMMI-75

## Commands to run DINOv1

```bash
python -m torch.distributed.launch --nproc_per_node={num_gpus} main_dino.py --arch {vit_small or vit_base or vit_large} --data_path /scr/data/75ds_small_train.zip {data_path} --output_dir /scr/vidit/Models/Dino_Base_75ds_Multiscale --lr 0.00005 --batch_size_per_gpu {batch_size}
```

## Commands to run MAE

```bash
python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py --data_path /scr/data/CHAMMIv2s_train.zip --output_dir /scr/vidit/Models/MAE_75ds_baseline --batch_size 1024
```
