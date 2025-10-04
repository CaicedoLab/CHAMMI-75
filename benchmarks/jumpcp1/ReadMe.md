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

