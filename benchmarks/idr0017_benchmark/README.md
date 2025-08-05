## IDR0017 Benchmark

This benchmarking is based on the paper "A chemical–genetic interaction map of small
molecules using high-throughput imaging in
cancer cells". The framework performs statistical analysis on the extracted feature embeddings of idr0017 and calculate scores for different cell lines present in the study. 

### ROC Test: 
Performs statistical analysis (effect size) for each cell line in the study and ranks all the compounds based on the effect size. ROC scores are used to check how well the model embeddings are able to rank HITs above Non-HITs.

### Fusion Type:
Determines how the replicates data is fused. Early fusion combines the replicates initially and then performs the statistical analysis. Late fusion perform analysis at replicate level and then combines the score at replicate level.


#### Entry point for the library: idr0017_benchmark.py
#### Configure the benchmark test: config.yaml