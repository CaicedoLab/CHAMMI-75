
## CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 extraction.py --model vit --model_path /scr/vidit/DINO_CHAMMI-75_LARGE_DATASET/checkpoint.pth --output_folder /scr/vidit/label-free-features/iclr_model --image_folder /scr/vidit/rbc-mc/

## python regression.py --output_folder /scr/vidit/label-free-features/iclr_model --pkl_path /scr/vidit/label-free-features/iclr_model/embeddings.pkl