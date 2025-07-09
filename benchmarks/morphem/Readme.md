# CHAMMI Evaluation

Code has been borrowed from https://github.com/broadinstitute/MorphEm and modified for evaluation purposes.

## Feature extraction for CHAMMI Scores

```bash
python feature_extraction.py --root_dir /scr/vidit/Foundation_Models/CHAMMI --feat_dir /scr/vidit/Foundation_Models/features_hpa --model vit --gpu 1 --batch_size 16

```

## Evaluating CHAMMI Score

```bash
python -c "from morphem.benchmark import run_benchmark; run_benchmark('/scr/vidit/Foundation_Models/CHAMMI', '/scr/vidit/Foundation_Models/CHAMMI_scores/Dino_Small_75ds_Guided', '/scr/vidit/Foundation_Models/features_Dino_Small_75ds_Guided', 'pretrained_vit_features.npy')"

```



