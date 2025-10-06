cp sc_inference.py /home/runner/sc_inference.py
cp sc_dataset_openphenom.py /home/runner/sc_dataset_openphenom.py

mkdir /home/runner/cache
export TRANSFORMERS_CACHE='/home/runner/OpenPhenom'

cp vit_encoder.py /home/runner/OpenPhenom/vit_encoder.py
cp huggingface_mae.py /home/runner/OpenPhenom/huggingface_mae.py

cp source4_$1_deepprofiler_single_cell_crops.zip $1.zip
unzip $1.zip

python3 /home/runner/sc_inference.py $1
