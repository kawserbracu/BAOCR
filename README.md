# BAOCR: Bengali-English OCR Thesis Pipeline

End-to-end pipeline to convert Label Studio annotations to DocTR-like data, preprocess images, train detection and recognition models, evaluate, and aggregate results.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
python run_full_pipeline.py \
  --labelstudio_json path/to/labelstudio.json \
  --images_dir path/to/images_dir \
  --output_base path/to/output_dir
```

This runs:
1. `data/analyze_dataset.py` – analyze dataset and produce summary + CSVs.
2. `data/convert_dataset.py` – convert LS to detection/recognition formats and split (70/15/15).
3. `data/preprocess.py` – generate original/clahe/highboost variants.
4. `training/train_detection.py` – template training (max_epochs=100, patience=15, adaptive LR).
5. `training/train_recognition.py` – CRNN training with CTC loss.
6. `evaluation/evaluate_end_to_end.py` – test 3x3 model combinations.
7. `evaluation/generate_results.py` – aggregate tables and plots.

## Directory Overview

- `data/analyze_dataset.py` – robust LS analyzer (classic + compact schema).
- `data/convert_dataset.py` – detection/recognition conversion + splits.
- `data/preprocess.py` – original/clahe/highboost preprocessing.
- `models/detection.py` – DocTR DB MobileNetV3 Small builder.
- `models/recognition.py` – CRNN (VGG16_bn + BiLSTM).
- `training/trainer_base.py` – early stopping, ReduceLROnPlateau, checkpoints, logs, curves.
- `training/train_detection.py` – template detection training loop.
- `training/train_recognition.py` – CRNN training loop with CTC.
- `evaluation/metrics.py` – metrics utilities (IoU, mAP approx, CER, WER).
- `evaluation/evaluate_detection.py` – evaluate detection on test split.
- `evaluation/evaluate_recognition.py` – evaluate recognition on test crops.
- `evaluation/evaluate_end_to_end.py` – end-to-end 9 combinations.
- `evaluation/generate_results.py` – aggregate comparison tables and plots.
- `inference/ocr_pipeline.py` – end-to-end OCR pipeline class.

## Notes
- Label Studio exports may contain server-like paths (e.g., `/data/upload/...`). Converters and analyzers resolve images by basename against your local `images_dir` and common split folders.
- Detection trainer here is a scaffold; DocTR predictors are inference-focused. Replace the dummy loss with proper DB loss if you plan to fine-tune.
- Recognition training adheres to faculty constraints: `max_epochs=100`, `patience=15`, and adaptive LR via ReduceLROnPlateau.
