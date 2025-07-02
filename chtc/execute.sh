export HOME=`pwd`? 
cd dinov1/
python -m torch.distributed.launch --nproc_per_node=8 main_dino.py --arch vit_small --data_path /scratch/CHAMMI10ds.zip --output_dir /hdd/jcaicedo/projects/CHAMMI/vidit_model_outputs/10ds_guided --lr 0.0005 --batch_size_per_gpu 224
# python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py --data_path /scr/data/CHAMMI10ds.zip --output_dir /scr/vidit/Models/MAE_10ds_baseline
