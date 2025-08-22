## IDR0017 Benchmark

This repository provides a benchmarking framework based on the study:  
**_A chemical–genetic interaction map of small molecules using high-throughput imaging in cancer cells_**.  

The framework performs statistical analysis on extracted feature embeddings from the **idr0017** dataset and computes scores for various cell lines included in the study.


### Instructions for Feature Extraction ###

To run this pipeline, following environment will need to be created.


Environment Creation:
``` shell
conda env create -f environment.yaml
conda activate weakly-supervised
```

To run the pipeline, we specify a configfile and a number of cores. The config file should point somewhere into either configs or local_configs depending on if you are running your study locally or not. 


``` shell
cd workflow
snakemake --configfile ../{local or chtc}_configs/your_config.yaml --jobs n --cores n*num_cores
```
Note: cores should be the amount of TOTAL cores required to run the pipeline. Jobs should be the number of GPUs used to run the pipeline.


### Parameters in config

To start, edit your config for your study with changes on the cache, study and metadata. Here are brief descriptions of the name and type `[]` of each parameter in the config file, as they apply per rule. 

#### For all rules we have these function:
- `out_folder[str]` Please use the given test set package and follow similar paths to set out_folder.
- `cache[str]` This is the local directory where the plate will be copied and unzipped at. This should be somewhere where there is some space.

For the next 3 entries, see the example DINOv2 config file for examples.

- `study[str]` Please use the given test set package and follow similar paths to set out_folder.
- `metadata[str]` Please use the given test set package and follow similar paths to set out_folder.
- `models[str]` The path to the cached model directory.


#### In the `resources` subsection we have:
- `num_gpus[int]` selects how many GPUs to use. This should match how many jobs you want to run concurrently when running locally. When running locally scale the --cores argument with threads so this runs this many jobs at once for ideal performance.
- `num_workers[int]` defines the num_workers argument for the data loader. 
- `threads[int]` is the number of CPU threads (read, cores) that will be available per each job run in this rule.
---

### In the `feature extraction` subsection we have:
- `crop` - crops of the single cell crops
- `resize` - resizes the crops for the model

Note: For adding your own models to the feature extraction snakemake pipeline, you can add your own model in model.py following the convention followed by the DINOv2 feature extraction pipeline.

### **ROC Test**
The ROC test evaluates the ability of model embeddings to distinguish **HITs** from **Non-HITs**.

**Workflow:**
1. For each cell line, compute statistical effect sizes for all compounds.
2. Rank compounds based on their effect sizes.
3. Calculate ROC scores to measure how well HITs are prioritized above Non-HITs.

---

### **Fusion Types**
Fusion type determines how replicate data is aggregated before or after statistical analysis:

- **Early Fusion** – Replicates are merged **before** performing statistical analysis.  
- **Late Fusion** – Statistical analysis is run **per replicate**, and scores are then combined.

---

### **How to Run**

**Entry point:**  
```bash
python idr0017_benchmark.py
