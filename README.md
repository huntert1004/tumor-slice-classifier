# Tumor Slice Classification (TensorFlow, BraTS24-Lite)

This repository contains a complete training and evaluation pipeline for MRI tumor slice classification using a 2D CNN. The project uses the BraTS24-Lite dataset, which must be downloaded separately due to size restrictions.

## Project Structure

```
tumor_detection/
│
├── brats24_lite/              # ⚠️ Not included — download required
├── models/                    # Saved .h5 trained models (v1.1 included)
├── results/                   # Evaluation plots and metrics
│
├── README.md
├── requirements.txt
│
├── train.py                   
├── run_evaluate_model.py      
├── run_evaluate_streaming.py  
```

## Dataset Download

The training script will automatically download BraTS24-Lite:

```python
snapshot_download(
    repo_id="YongchengYAO/BraTS24-Lite",
    repo_type="dataset",
    local_dir="brats24_lite"
)
```

Just run:

```bash
python train.py
```

## Preprocessing

The script:
- Loads each 3D MRI volume
- Normalizes intensities
- Slices into 2D 128×128 axial images
- Assigns tumor/non‑tumor labels
- Saves each volume as an NPZ in `preprocessed_volumes/`

## Training (Streaming)

Train the improved streaming model:

```bash
python train.py
```

## Evaluating Model v1.1

```bash
python run_evaluate_streaming.py
```

This generates:
- True/False positive/negative slices
- ROC curve
- Classification report
- Confusion matrix

Outputs save to:

```
results/
```

## Python Version

**Python 3.10+ recommended**
