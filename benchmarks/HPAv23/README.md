# HPA Sub-Cell Evaluation

## Overview
This folder contains code for running HPA (Human Protein Atlas) features extraction and protein localization evaluations on the mini-HPA dataset.

---

## HPA Features Extraction

### Command
```bash
accelerate launch --multi_gpu --num_processes=8 accelerate_hpa_features.py \
  --model {type of model} \
  --model_path {model paths} \
  --model_size {model sizes} \
  --image_folder {pointing to mini-hpa} \
  --batch_size {batch_size} \
  --num_workers {how many workers loaded} \
  --output_folder {output folder for features}
```

### Flag Documentation

#### Accelerate Launcher Flags
- `--multi_gpu`: Enable multi-GPU processing for distributed training/inference
- `--num_processes=8`: Number of GPU processes to use for parallel computation (adjust based on available GPUs)

#### Model Configuration Flags
- `--model`: Type of model architecture to use for feature extraction (e.g., resnet50, vit, etc.)
- `--model_path`: Path(s) to the pretrained model weights or checkpoint file(s). Can be a single path or multiple paths separated by commas.
- `--model_size`: Size or variant of the model (e.g., small, base, large). Must correspond with the specified model type.

#### Data Processing Flags
- `--image_folder`: Path to the mini-HPA dataset directory containing the input images. Should point to the root directory of the mini-HPA dataset.
- `--batch_size`: Number of images to process per batch. Adjust based on available GPU memory (typical values: 32, 64, 128).
- `--num_workers`: Number of worker processes for data loading. Increase for faster data pipeline, but monitor CPU/memory usage (typical values: 4, 8, 16).

#### Output Flag
- `--output_folder`: Directory path where extracted features will be saved. Creates directory if it doesn't exist. Features will be stored as feature matrices or embedding files.

### Notes
- **Feature extraction hanging**: The code may hang after features are saved. You may need to manually terminate the processes using `Ctrl+C` or keyboard interrupts if the script doesn't exit cleanly.
- **GPU memory considerations**: If you encounter out-of-memory errors, reduce `--batch_size` or `--num_processes`.
- **Data loading optimization**: Adjust `--num_workers` based on your system's CPU cores and available RAM.

---

## Protein Localization Evaluation

### Commands

#### Evaluation on all unique categories based on SubCell paper categories:
```bash
python train_classification.py \
  -f {saving_metrics_locations} \
  -cc locations \
  -uc all_unique_cats
```

#### Evaluation on challenge categories based on the Kaggle classification challenge:
```bash
python train_classification.py \
  -f {saving_metrics_locations} \
  -cc locations \
  -uc challenge_cats
```

### Flag Documentation

- `-f {saving_metrics_locations}`: Path to the folder where evaluation metrics and results will be saved. **This must be the same folder where features were extracted in the features extraction step.** The script will look for previously extracted features in this location.

- `-cc locations`: Classification category. Specifies that you are performing protein localization classification. (Fixed value for this use case)

- `-uc {category_type}`: Unique categories to evaluate on
  - `all_unique_cats`: Evaluates on all unique protein localization categories present in the dataset (23 locations)
  - `challenge_cats`: Evaluates only on the challenge subset of protein localization categories (19 locations)

### Important Notes
- **Feature location matching**: The `-f` flag must point to the exact folder where features were extracted by `accelerate_hpa_features.py`. The script expects to find feature files in this directory.
- **Output location**: Metrics, logs, and results will be saved in the same folder specified by `-f`.
- **Running both evaluations**: You can run both commands sequentially to compare performance on all categories versus challenge categories.

---

## Workflow Example

A typical workflow would look like:

```bash
# Step 1: Extract features using multi-GPU processing
accelerate launch --multi_gpu --num_processes=8 accelerate_hpa_features.py \
  --model resnet50 \
  --model_path ./pretrained_weights/model.pth \
  --model_size base \
  --image_folder /path/to/mini-hpa \
  --batch_size 128 \
  --num_workers 8 \
  --output_folder /path/to/features_output

# Step 2: Evaluate on all categories
python train_classification.py \
  -f /path/to/features_output \
  -cc locations \
  -uc all_unique_cats

# Step 3: Evaluate on challenge categories
python train_classification.py \
  -f /path/to/features_output \
  -cc locations \
  -uc challenge_cats
```

---
