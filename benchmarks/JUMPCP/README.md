# Jump-CP Evaluation

## Overview
This folder contains all the required metadata and code to run the Jump-CP evaluation benchmark.

##
 Feature Extraction from Jump-CP Images

###
 Command

```
bash
python feature_extraction.py
  \ --model vit 
  \ --model_path /scr/vidit/DINO_CHAMMI-75_LARGE_DATASET/checkpoint.pth 
  \ --root_dir /scr/data/CHAMMI-75_test/Jump-CP/ 
  \ --feat_dir /scr/vidit/jump_results/ 
  \ --batch_size 128
```

### Flag Documentation

- `--model`: Type of model architecture to use for feature extraction (e.g., `vit` for Vision Transformer). Specifies which model backbone will generate the feature embeddings.
- `--model_path`: Path to the pretrained model checkpoint/weights file. This should be a `.pth` file containing the model weights. Example: `/scr/vidit/DINO_CHAMMI-75_LARGE_DATASET/checkpoint.pth`
- `--root_dir`: Path to the root directory of the Jump-CP dataset. This directory should contain all the image files and metadata for the Jump-CP compound screening dataset.
- `--feat_dir`: Output directory where extracted features will be saved. The script will create feature files (typically in `.npy` or `.pt` format) in this location. Create this directory beforehand if it doesn't exist.
- `--batch_size`: Number of images to process per batch during feature extraction (typical values: 64, 128, 256). Adjust based on available GPU memory—reduce if you encounter out-of-memory errors.

### Output

The script generates feature files containing embeddings extracted from the model. These features are saved in `--feat_dir` and used in subsequent aggregation and evaluation steps.

---

## Well-Level Profile Aggregation

### Command

```
bash
python well_level_aggregation.py 
 \ --model dinov1 
 \ --profiles /scr/vidit/jump_results/ 
 \ --output path_to_output
```

### Flag Documentation

- `--model`: Model variant identifier for aggregation purposes (e.g., `dinov1` for DINO v1, `vit`, etc.). This identifies which model's features are being aggregated and should match the model used in feature extraction.
- `--profiles`: Path to the directory containing extracted features from the previous feature extraction step. This should match the 
`--feat_dir` used in `feature_extraction.py`. The script will read feature files from this directory.
- `--output`: Directory path where aggregated well-level profiles will be saved. This directory will contain the aggregated feature representations at the well level (combining features from multiple images per well). Create this directory beforehand if it doesn't exist.

### Purpose

Well-level aggregation combines per-image features into representative profiles for each well in the plate. This aggregation step pools information across multiple images within a well to create robust cellular phenotype representations.

---

## Evaluation on Jump-CP Benchmark

### Command

```
bash
python run_evaluation.py \ --model dinov1 \ --output path_to_output
```

### Flag Documentation

- `--model` : Model variant identifier being evaluated (e.g., `dinov1`). Should match the model identifier used in the aggregation step. This is used for tracking and reporting results.
- `--output`: Path to the directory containing the aggregated well-level profiles from the previous aggregation step. The evaluation script will read the profiles from this location and compute benchmark metrics.

### Purpose

This script evaluates the quality of the aggregated profiles using Jump-CP benchmark metrics. It computes various performance measures that assess how well the model's features capture biologically meaningful information from the compound screening data.


### Troubleshooting Tips


Data (Cellprofiler profiles and metadata) are from Chandrasekaran et al. 2024 Nature Methods paper, [original repository](https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods/). 

Evaluation pipeline is based on `copairs` method from Kalinin et al. 2025 Nature Communications.
Install `copairs` version `0.5.1`. 

Single-cell crops are located in Data Vault in `cellpainting-data/cpj1_single_cell_crops/` folder. 
Well-level profiles are expected to be in `features` folder. 

- `prepare_and_download.sh` - download CellProfiler profiles from Chandrasekaran et al. 2024 as a reference

- `feature_extraction.py` - works with DINOv2 and DINOv3 pretrained models.
- `sc_dataset` - data loader of single-cells. Single-cell crops are prepared with DeepProfiler.  
- `well_level_aggregation.py` - feature aggregation (mean profile of cells within the well) and batch correction. See the dicts inside for feature dimensionality.
- `run_evaluation.py` - run benchmarks, results would appear in `results/{model}` folder. 


SubCell feature extraction - separate code was used:
- images were center-croped 128x128 and then resized to 256x256 
- [example code and repository](https://github.com/CaicedoLab/SubCell_CellPainting)

OpenPhenom feature extraction, code in `openphenom_chtc`
- Docker image was used with pre-installed environment and code from [original repository](https://huggingface.co/recursionpharma/OpenPhenom/)
- CHTC cluster was used for feature extraction

