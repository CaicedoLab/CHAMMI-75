# CELLPHIE Evaluation

## Overview
CELLPHIE is a 14-channel cellular morphology evaluation benchmark for foundation models.

---

## Feature Extraction

### Command
```bash
accelerate launch --multi_gpu --num_processes=7 extraction.py \
  --model dinov2 \
  --image_folder /scr/vidit/neural_features/ \
  --output_folder /scr/vidit/neural_features/neuron_feature_extraction/DINOv2/ \
  --num_workers 8
```

### Flag Documentation

- `--multi_gpu`: Enable multi-GPU distributed processing
- `--num_processes=7`: Number of GPU processes (match available GPUs)
- `--model`: Foundation model architecture (e.g., `dinov2`, `vit`)
- `--image_folder`: Path to 14-channel image data directory
- `--output_folder`: Directory where extracted features will be saved
- `--num_workers`: Number of parallel data loading workers

---

## Gene Knockout Classification

### Command
```bash
python classifier.py \
  --embedding_path /scr/vidit/neural_features/neuron_feature_extraction/DINOv2/
  --embed_dim 384
```

### Flag Documentation

- `--embedding_path`: Path to the directory containing extracted features (must match `--output_folder` from feature extraction step)
- `--embed_dim`: Number of dimensions of the output given by the model being tested. Used to decide bounds for the PCA which help in evaluating the 14 channel study

---

## Complete Workflow

```bash
# Step 1: Extract features from 14-channel images
accelerate launch --multi_gpu --num_processes=7 extraction.py \
  --model dinov2 \
  --image_folder /scr/vidit/neural_features/ \
  --output_folder /scr/vidit/neural_features/neuron_feature_extraction/DINOv2/ \
  --num_workers 8

# Step 2: Train and evaluate phenotype classifier
python classifier.py \
  --embedding_path /scr/vidit/neural_features/neuron_feature_extraction/DINOv2/
```
