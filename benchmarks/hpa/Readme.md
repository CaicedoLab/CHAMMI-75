For running hpa features extraction


accelerate launch --multi_gpu --num_processes=8 accelerate_hpa_features.py


For running cell_line evaluation in HPAv23

python train_classification.py -f {saving_metrics_locations} -cc atlas_name

For running locations evaluation in HPAv23

python train_classification.py -f {saving_metrics_locations} -cc locations -uc all_unique_cats

