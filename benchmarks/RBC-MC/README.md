# RBC-MC Classification

## Overview
RBC-MC is a cross-domain benchmark for classifying red blood cell (RBC) phenotypes. This evaluation assesses foundation models' ability to transfer learned representations to morphological classification of RBCs across different imaging domains.

---

## Feature Extraction

### Command
```bash
accelerate launch --num_processes=1 extraction.py \
  --model vit \
  --model_path /scr/vidit/DINO_CHAMMI-75_LARGE_DATASET/checkpoint.pth \
  --output_folder /scr/vidit/label-free-features/iclr_model \
  --image_folder /scr/vidit/rbc-mc/
```

### Flag Documentation

- `--num_processes=1`: Number of processes for accelerate (set to 1 for single GPU processing).

- `--model`: Foundation model architecture to use for feature extraction (e.g., `vit` for Vision Transformer).

- `--model_path`: Path to the pretrained model checkpoint. Should point to the saved weights file (`.pth` format).

- `--output_folder`: Directory where extracted RBC features will be saved. Features will be stored as embeddings for downstream regression.

- `--image_folder`: Path to the RBC-MC dataset directory containing RBC images across different domains.

---

## Phenotype Regression

### Command
```bash
python regression.py \
  --output_folder /scr/vidit/label-free-features/iclr_model \
  --pkl_path /scr/vidit/label-free-features/iclr_model/embeddings.pkl
```

### Flag Documentation

- `--output_folder`: Directory where regression results and predictions will be saved. Should match the `--output_folder` from feature extraction.

- `--pkl_path`: Path to the embeddings pickle file containing extracted features. Typically located in the feature extraction output folder as `embeddings.pkl`. Must match the location where features were saved.

---

## Complete Workflow

```bash
# Step 1: Extract features from RBC-MC images
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 extraction.py \
  --model vit \
  --model_path /scr/vidit/DINO_CHAMMI-75_LARGE_DATASET/checkpoint.pth \
  --output_folder /scr/vidit/label-free-features/iclr_model \
  --image_folder /scr/vidit/rbc-mc/

# Step 2: Perform phenotype regression on extracted features
python regression.py \
  --output_folder /scr/vidit/label-free-features/iclr_model \
  --pkl_path /scr/vidit/label-free-features/iclr_model/embeddings.pkl
```

### Important Notes
- **Path consistency**: The `--output_folder` from extraction must match the `--output_folder` in regression
- **Embeddings file**: Ensure `embeddings.pkl` is generated in the extraction step and available at the `--pkl_path` location
- **Cross-domain evaluation**: RBC-MC tests model generalization across different imaging domains
