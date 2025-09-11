# HPA Sub-Cell Evaluation

## For running hpa features extraction

```
accelerate launch --multi_gpu --num_processes=8 accelerate_hpa_features.py --model {type of model} --model_path {model paths} --model_size {model sizes} --image_folder {pointing to mini-hpa} --batch_size {batch_size} --num_workers {how many workers loaded} --config_path {used only for subcell} --output_folder {output folder for features}
```

Note: Feature extraction code hangs at the end after features save. Might need to kill the processes on your own after that using keyboard interrupts

## For running protein localization evaluations in mini-hpa

```
python train_classification.py -f {saving_metrics_locations} -cc locations -uc all_unique_cats

python train_classification.py -f {saving_metrics_locations} -cc locations -uc challenge_cats
```

Note: Saving_metrics_locations needs to be the same folder where the features were extracted out!
