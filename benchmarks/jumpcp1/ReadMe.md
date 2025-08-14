 
Metrics and calculation code is from Chandrasekaran et al. 2024 Nature Methods paper, [original repository](https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods/). 

Create new environment with packages from `copairs_old_env.yml`, it is old version that was used in the original paper.

Single-cell crops are located in Data Vault in `cellpainting-data/cpj1_single_cell_crops/` folder. 
Well-level profiles are expected to be in `features` folder. 

`misc/well_level_aggregation_example.py` - feature aggregation example for DINO. In principle the same applies to other deep learning methods (mind the feature dimensionality). 

Results for reporting would be in `fr_replicability_{feature_extractor}_results.csv` and `fr_matching_{feature_extractor}_results.csv`. 


Feature conversion example in `misc/convert_features.ipynb`