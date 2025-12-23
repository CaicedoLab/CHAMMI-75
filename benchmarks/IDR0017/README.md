## IDR0017 Benchmark

This repository provides a benchmarking framework based on the study:  
**_A chemical–genetic interaction map of small molecules using high-throughput imaging in cancer cells_**.  

The framework performs statistical analysis on extracted feature embeddings from the **idr0017** dataset and computes scores for various cell lines included in the study.

### Command for feature extraction

```bash 
python feature_extraction.py --model_path /scr/vidit/DINO_CHAMMI-75_LARGE_DATASET/checkpoint.pth --model_type vit --batch_size 2048 --images_folder /scr/data/idr0017 --output_folder /scr/vidit/idr17_features/dino --num_workers 4
```

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
