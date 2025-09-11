# Neuron Evaluation

```bash
accelerate launch --multi_gpu --num_processes=7 extraction.py --model dinov2 --image_folder /scr/vidit/neural_features/ --output_folder /scr/vidit/neural_features/neuron_feature_extraction/DINOv2/ --num_workers 8
```