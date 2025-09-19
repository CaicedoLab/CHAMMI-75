. /etc/profile.d/pixi.sh 
unzip ./chammi_dataset.zip
cd ./CHAMMI-75/benchmarks/morphem/
python feature_extraction.py --root_dir ../../../dataset --feat_dir $FEATURE_DIR --model $MODEL_TYPE --model_size $MODEL_SIZE --model_path $MODEL_PATH  --batch_size 12   
