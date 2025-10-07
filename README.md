# BAOCR: Bengali-English OCR Thesis Pipeline

End-to-end pipeline for Bengali-English handwritten text recognition. Converts Label Studio annotations to DocTR-like data, applies preprocessing (CLAHE, High-boost), trains detection and recognition models with attention mechanisms, and provides comprehensive evaluation.

## Dataset Overview

**Total Dataset**: 450 images (5 sources)
- **d1**: 85 images → `d1_annotations.json`
- **d2**: 47 images → `d2_annotations.json` 
- **d3**: 20 images → `d3_annotations.json`
- **d4**: 38 images → `d4_annotations.json`
- **raw**: 260 images → 260 individual `.json` files (merged)

**Preprocessing Variants**:
- **Original**: Base images
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization
- **High-boost**: High-boost filtering for edge enhancement

**Final Combined Dataset**: 59,457 samples (Original + CLAHE + High-boost)
- **Training**: 42,780 samples
- **Validation**: 8,177 samples  
- **Test**: 8,500 samples

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
## Architecture Overview

### **Detection Models**
- **3 separate models**: Original, CLAHE, High-boost variants
- **Architecture**: Custom MobileNetV3 + DB-like segmentation
- **Training**: BCE+Dice loss, 100 epochs, patience=15

### **Recognition Model (Final)**
- **Architecture**: VGG16_bn + Reduced BiLSTM + MultiheadAttention
- **Optimizations**: 
  - BiLSTM: 1 layer, 128 hidden units (reduced complexity)
  - Attention: 4 heads, 256 embed_dim
  - Image scaling: H=128 (4x increase from 32)
- **Training**: Combined dataset (59K samples), CTC loss + label smoothing
- **Data Augmentation**: Rotation, elastic transform, brightness/contrast, blur, noise
- **Parameters**: LR=2e-4, batch=12, 200 epochs, patience=25, gradient clipping=5.0

### **Key Components**
- `data/` – Dataset conversion, preprocessing, merging utilities
- `models/` – Detection (MobileNetV3) and Recognition (CRNN+Attention) architectures  
- `training/` – Training loops with early stopping, checkpointing, smart cleanup
- `evaluation/` – Comprehensive metrics (IoU, mAP, CER, WER) and visualization
- `tools/` – Dataset combination, fine-tuning utilities
- `inference/` – End-to-end OCR pipeline

## Faculty Requirements Implementation

✅ **All faculty suggestions implemented**:
- **Image scaling**: 32 → 128 pixels height (4x increase)
- **Preprocessing**: CLAHE + High-boost filtering variants
- **Data augmentation**: Enhanced for handwritten Bengali text
- **Attention mechanism**: MultiheadAttention in CRNN
- **Training parameters**: 
  - Batch size: 12 (optimized for combined dataset)
  - Max epochs: 200
  - Patience: 25  
  - Gradient clipping: max_norm=5.0
  - Label smoothing: 0.1 for CTC loss
- **Model optimization**: Reduced layers + increased data (10x dataset size)
- **Smart checkpointing**: Keep only 3 recent + best model (98% storage reduction)
## Detection: inference (DocTR MobileNetV3)
Uses DocTR predictor (downloads pretrained weights). MobileNetV3 Large is preferred, Small as fallback.

```powershell
python "c:\Users\T2510600\Downloads\BAOCR\inference\ocr_pipeline.py" --images_dir "c:\Users\T2510600\Downloads\BAOCR\d4" --rec_model "c:\Users\T2510600\Downloads\BAOCR\out\merged\runs\rec_run\model_best.pt" --vocab "c:\Users\T2510600\Downloads\BAOCR\out\merged\vocab.json" --output_dir "c:\Users\T2510600\Downloads\BAOCR\out\merged\inference_d4"
```

## Detection: training (Faster R-CNN fallback)
Trains a standard detector with your constraints (100 epochs, patience=15, ReduceLROnPlateau).

```powershell
python "c:\Users\T2510600\Downloads\BAOCR\training\train_detection_db.py" --data_dir "c:\Users\T2510600\Downloads\BAOCR\out\merged" --arch "db_mobilenet_v3_small" --output_name "det_db_run" --batch_size 4
```

Evaluate (DocTR-style eval script on test manifest):

```powershell
python "c:\Users\T2510600\Downloads\BAOCR\evaluation\evaluate_detection.py" --test_data "c:\Users\T2510600\Downloads\BAOCR\out\merged\detection_test.json" --output_dir "c:\Users\T2510600\Downloads\BAOCR\out\merged\eval_det_trained"
```

## Detection: training (Custom MobileNetV3 DB-like)
Trains a segmentation-style detector (MobileNetV3 Large/Small backbone + light decoder, BCE+Dice) and converts masks to boxes.

```powershell
python "c:\Users\T2510600\Downloads\BAOCR\training\train_detection_db_custom.py" --data_dir "c:\Users\T2510600\Downloads\BAOCR\out\merged" --backbone large --output_name "det_db_mnv3_custom_large" --batch_size 4
```

Evaluate the custom detector:

```powershell
python "c:\Users\T2510600\Downloads\BAOCR\evaluation\evaluate_custom_db.py" --model "c:\Users\T2510600\Downloads\BAOCR\out\merged\runs\det_db_mnv3_custom_large\model_best.pt" --backbone large --test_data "c:\Users\T2510600\Downloads\BAOCR\out\merged\detection_test.json" --output_dir "c:\Users\T2510600\Downloads\BAOCR\out\merged\eval_det_custom"
```

Use the custom detector in end-to-end OCR:

```powershell
python "c:\Users\T2510600\Downloads\BAOCR\inference\ocr_pipeline.py" --images_dir "c:\Users\T2510600\Downloads\BAOCR\d4" --rec_model "c:\Users\T2510600\Downloads\BAOCR\out\merged\runs\rec_run\model_best.pt" --vocab "c:\Users\T2510600\Downloads\BAOCR\out\merged\vocab.json" --output_dir "c:\Users\T2510600\Downloads\BAOCR\out\merged\inference_d4_custom" --custom_det_model "c:\Users\T2510600\Downloads\BAOCR\out\merged\runs\det_db_mnv3_custom_large\model_best.pt" --custom_backbone large --mask_threshold 0.35 --limit 20
```

## CLAHE comparison and report

Rebase test manifest to CLAHE images, evaluate, and generate combined report.

```powershell
python "c:\Users\T2510600\Downloads\BAOCR\data\rebase_manifest.py" --input_json "c:\Users\T2510600\Downloads\BAOCR\out\merged\detection_test.json" --new_root "c:\Users\T2510600\Downloads\BAOCR\preproc\clahe\images" --output_json "c:\Users\T2510600\Downloads\BAOCR\out\merged\detection_test_clahe.json"
python "c:\Users\T2510600\Downloads\BAOCR\evaluation\evaluate_custom_db.py" --model "c:\Users\T2510600\Downloads\BAOCR\out\merged\runs\det_db_mnv3_custom_large\model_best.pt" --backbone large --test_data "c:\Users\T2510600\Downloads\BAOCR\out\merged\detection_test_clahe.json" --output_dir "c:\Users\T2510600\Downloads\BAOCR\out\merged\eval_det_custom_clahe" --mask_threshold 0.35
python "c:\Users\T2510600\Downloads\BAOCR\evaluation\generate_report.py" --rec_eval "c:\Users\T2510600\Downloads\BAOCR\out\merged\eval_rec\recognition_eval.json" --det_eval "doctr_mnv3_large=c:\Users\T2510600\Downloads\BAOCR\out\merged\eval_det\detection_eval.json" "custom_mnv3_large=c:\Users\T2510600\Downloads\BAOCR\out\merged\eval_det_custom\detection_eval_custom.json" "custom_mnv3_large_clahe=c:\Users\T2510600\Downloads\BAOCR\out\merged\eval_det_custom_clahe\detection_eval_custom.json" --output_dir "c:\Users\T2510600\Downloads\BAOCR\out\merged\report"
```

---

## Reproduced workflow (latest, multi-model matrix)

This section captures the exact steps we used to build merged data, train 3×2 models (3 detection × 3 recognition across Original/CLAHE/High-boost), and generate evaluations and figures.

### 1) Convert datasets to DocTR manifests

- d1–d4 from Label Studio
```powershell
python "c:\Users\T2510600\Downloads\BAOCR\data\convert_dataset.py" --labelstudio_json "c:\Users\T2510600\Downloads\BAOCR\d1_annotations.json" --images_dir "c:\Users\T2510600\Downloads\BAOCR\d1" --output_dir "c:\Users\T2510600\Downloads\BAOCR\out\d1" --vocab_path "c:\Users\T2510600\Downloads\BAOCR\out\merged_all\vocab.json"; python "c:\Users\T2510600\Downloads\BAOCR\data\convert_dataset.py" --labelstudio_json "c:\Users\T2510600\Downloads\BAOCR\d2_annotations.json" --images_dir "c:\Users\T2510600\Downloads\BAOCR\d2" --output_dir "c:\Users\T2510600\Downloads\BAOCR\out\d2" --vocab_path "c:\Users\T2510600\Downloads\BAOCR\out\merged_all\vocab.json"; python "c:\Users\T2510600\Downloads\BAOCR\data\convert_dataset.py" --labelstudio_json "c:\Users\T2510600\Downloads\BAOCR\d3_annotations.json" --images_dir "c:\Users\T2510600\Downloads\BAOCR\d3" --output_dir "c:\Users\T2510600\Downloads\BAOCR\out\d3" --vocab_path "c:\Users\T2510600\Downloads\BAOCR\out\merged_all\vocab.json"; python "c:\Users\T2510600\Downloads\BAOCR\data\convert_dataset.py" --labelstudio_json "c:\Users\T2510600\Downloads\BAOCR\d4_annotations.json" --images_dir "c:\Users\T2510600\Downloads\BAOCR\d4" --output_dir "c:\Users\T2510600\Downloads\BAOCR\out\d4" --vocab_path "c:\Users\T2510600\Downloads\BAOCR\out\merged_all\vocab.json"
```

- RAW per-image JSONs (optional)
```powershell
python "c:\Users\T2510600\Downloads\BAOCR\data\convert_raw_per_image.py" --raw_dir "c:\Users\T2510600\Downloads\BAOCR\raw" --output_dir "c:\Users\T2510600\Downloads\BAOCR\out\raw"
```

### 2) Merge all splits (build fresh train/val/test)
```powershell
python -c "import json,random,os; from pathlib import Path; base=Path(r'c:\Users\T2510600\Downloads\BAOCR'); outs=[base/'out'/'d1',base/'out'/'d2',base/'out'/'d3',base/'out'/'d4',base/'out'/'raw']; items=[]; rec_all=[]; [items.extend(json.loads((od/'detection_all.json').read_text(encoding='utf-8')).get('items',[])) for od in outs if (od/'detection_all.json').exists()]; [rec_all.extend(json.loads((od/'recognition_all.json').read_text(encoding='utf-8')).get('items',[])) for od in outs if (od/'recognition_all.json').exists()]; out_dir=base/'out'/'merged_all'; out_dir.mkdir(parents=True,exist_ok=True); uniq=list({it['img_path'] for it in items}); random.seed(42); random.shuffle(uniq); n=len(uniq); ntr=int(round(n*0.7)); nva=int(round(n*0.15)); S={'train':set(uniq[:ntr]),'val':set(uniq[ntr:ntr+nva]),'test':set(uniq[ntr+nva:])}; filt=lambda k:[it for it in items if it['img_path'] in S[k]]; spl={k:filt(k) for k in ('train','val','test')}; [(out_dir/f'detection_{k}.json').write_text(json.dumps({'items':spl[k]},ensure_ascii=False,indent=2),encoding='utf-8') for k in spl]; from pathlib import Path as P; img2split={P(it['img_path']).stem:k for k in ('train','val','test') for it in spl[k]}; rec={'train':[],'val':[],'test':[]}; [rec[img2split.get(P(r['crop_path']).stem.rsplit('_word',1)[0],'train')].append(r) for r in rec_all if P(r['crop_path']).exists() and P(r['crop_path']).stem.rsplit('_word',1)[0] in img2split]; [(out_dir/f'recognition_{k}.json').write_text(json.dumps({'items':rec[k]},ensure_ascii=False,indent=2),encoding='utf-8') for k in ('train','val','test')]; voc=out_dir/'vocab.json'; voc.write_text(voc.read_text(encoding='utf-8') if voc.exists() else json.dumps({'chars':[]},ensure_ascii=False,indent=2),encoding='utf-8'); print('Merged to',out_dir)"
```

### 3) Create CLAHE/High-boost detection manifests
```powershell
mkdir "c:\Users\T2510600\Downloads\BAOCR\out\merged_all_clahe" -Force; python "c:\Users\T2510600\Downloads\BAOCR\data\rebase_manifest.py" --input_json "c:\Users\T2510600\Downloads\BAOCR\out\merged_all\detection_train.json" --new_root "c:\Users\T2510600\Downloads\BAOCR\preproc\clahe\images" --output_json "c:\Users\T2510600\Downloads\BAOCR\out\merged_all_clahe\detection_train.json"; python "c:\Users\T2510600\Downloads\BAOCR\data\rebase_manifest.py" --input_json "c:\Users\T2510600\Downloads\BAOCR\out\merged_all\detection_val.json" --new_root "c:\Users\T2510600\Downloads\BAOCR\preproc\clahe\images" --output_json "c:\Users\T2510600\Downloads\BAOCR\out\merged_all_clahe\detection_val.json"; python "c:\Users\T2510600\Downloads\BAOCR\data\rebase_manifest.py" --input_json "c:\Users\T2510600\Downloads\BAOCR\out\merged_all\detection_test.json" --new_root "c:\Users\T2510600\Downloads\BAOCR\preproc\clahe\images" --output_json "c:\Users\T2510600\Downloads\BAOCR\out\merged_all_clahe\detection_test.json"
```
```powershell
mkdir "c:\Users\T2510600\Downloads\BAOCR\out\merged_all_highboost" -Force; python "c:\Users\T2510600\Downloads\BAOCR\data\rebase_manifest.py" --input_json "c:\Users\T2510600\Downloads\BAOCR\out\merged_all\detection_train.json" --new_root "c:\Users\T2510600\Downloads\BAOCR\preproc\highboost\images" --output_json "c:\Users\T2510600\Downloads\BAOCR\out\merged_all_highboost\detection_train.json"; python "c:\Users\T2510600\Downloads\BAOCR\data\rebase_manifest.py" --input_json "c:\Users\T2510600\Downloads\BAOCR\out\merged_all\detection_val.json" --new_root "c:\Users\T2510600\Downloads\BAOCR\preproc\highboost\images" --output_json "c:\Users\T2510600\Downloads\BAOCR\out\merged_all_highboost\detection_val.json"; python "c:\Users\T2510600\Downloads\BAOCR\data\rebase_manifest.py" --input_json "c:\Users\T2510600\Downloads\BAOCR\out\merged_all\detection_test.json" --new_root "c:\Users\T2510600\Downloads\BAOCR\preproc\highboost\images" --output_json "c:\Users\T2510600\Downloads\BAOCR\out\merged_all_highboost\detection_test.json"
```

### 4) Build recognition crops/manifests for CLAHE/High-boost
```powershell
python -c "import json,sys; from pathlib import Path; sys.path.insert(0,r'c:\Users\T2510600\Downloads\BAOCR'); from data.convert_dataset import convert_recognition, save_split_recognition; od=Path(r'c:\Users\T2510600\Downloads\BAOCR\out\merged_all_clahe'); vt=od/'vocab.json'; tr=json.load(open(od/'detection_train.json',encoding='utf-8'))['items']; va=json.load(open(od/'detection_val.json',encoding='utf-8'))['items']; te=json.load(open(od/'detection_test.json',encoding='utf-8'))['items']; rec_all=convert_recognition(tr+va+te, od, vt); save_split_recognition(rec_all, tr, va, te, od); print('CLAHE recognition manifests ready:', od)"
```
```powershell
python -c "import json,sys; from pathlib import Path; sys.path.insert(0,r'c:\Users\T2510600\Downloads\BAOCR'); from data.convert_dataset import convert_recognition, save_split_recognition; od=Path(r'c:\Users\T2510600\Downloads\BAOCR\out\merged_all_highboost'); vt=od/'vocab.json'; tr=json.load(open(od/'detection_train.json',encoding='utf-8'))['items']; va=json.load(open(od/'detection_val.json',encoding='utf-8'))['items']; te=json.load(open(od/'detection_test.json',encoding='utf-8'))['items']; rec_all=convert_recognition(tr+va+te, od, vt); save_split_recognition(rec_all, tr, va, te, od); print('High-boost recognition manifests ready:', od)"
```

### 5) Train 3 detection models
```powershell
python "c:\Users\T2510600\Downloads\BAOCR\training\train_detection_db_custom.py" --data_dir "c:\Users\T2510600\Downloads\BAOCR\out\merged_all" --backbone large --output_name "det_db_mnv3_custom_large_original" --batch_size 4
```
```powershell
python "c:\Users\T2510600\Downloads\BAOCR\training\train_detection_db_custom.py" --data_dir "c:\Users\T2510600\Downloads\BAOCR\out\merged_all_clahe" --backbone large --output_name "det_db_mnv3_custom_large_clahe" --batch_size 4
```
```powershell
python "c:\Users\T2510600\Downloads\BAOCR\training\train_detection_db_custom.py" --data_dir "c:\Users\T2510600\Downloads\BAOCR\out\merged_all_highboost" --backbone large --output_name "det_db_mnv3_custom_large_highboost" --batch_size 4
```

### 6) Create Combined Dataset & Train Final Recognition Model
```powershell
# Combine all preprocessing variants for maximum data
python "c:\Users\T2510600\Downloads\BAOCR\tools\create_combined_dataset.py" --base_dir "c:\Users\T2510600\Downloads\BAOCR\out" --output_name "merged_all_combined" --datasets "merged" "merged_all_clahe" "merged_all_highboost"

# Train final optimized recognition model
python "c:\Users\T2510600\Downloads\BAOCR\training\train_recognition.py" --data_dir "c:\Users\T2510600\Downloads\BAOCR\out\merged_all_combined" --vocab_path "c:\Users\T2510600\Downloads\BAOCR\out\merged_all_combined\vocab.json" --output_name "rec_v3_reduced_combined_augplus"
```

### 7) Evaluate Final Models
```powershell
# Evaluate detection models (3 variants)
python "c:\Users\T2510600\Downloads\BAOCR\evaluation\evaluate_custom_db.py" --model "c:\Users\T2510600\Downloads\BAOCR\out\merged_all\runs\det_db_mnv3_custom_large_original\model_best.pt" --backbone large --test_data "c:\Users\T2510600\Downloads\BAOCR\out\merged_all\detection_test.json" --output_dir "c:\Users\T2510600\Downloads\BAOCR\out\merged_all\eval_det_custom" --mask_threshold 0.35

# Evaluate final recognition model (combined dataset)
python "c:\Users\T2510600\Downloads\BAOCR\evaluation\evaluate_recognition.py" --model_path "c:\Users\T2510600\Downloads\BAOCR\out\merged_all_combined\runs\rec_v3_reduced_combined_augplus\model_best.pt" --test_data "c:\Users\T2510600\Downloads\BAOCR\out\merged_all_combined\recognition_test.json" --vocab_path "c:\Users\T2510600\Downloads\BAOCR\out\merged_all_combined\vocab.json" --output_dir "c:\Users\T2510600\Downloads\BAOCR\out\merged_all_combined\eval_rec_v3_reduced"
```

### 8) Comparative plots
```powershell
python "c:\Users\T2510600\Downloads\BAOCR\evaluation\plot_detection_results.py" --det_eval "orig=c:\Users\T2510600\Downloads\BAOCR\out\merged_all\eval_det_custom\detection_eval_custom.json" "clahe=c:\Users\T2510600\Downloads\BAOCR\out\merged_all_clahe\eval_det_custom\detection_eval_custom.json" "highboost=c:\Users\T2510600\Downloads\BAOCR\out\merged_all_highboost\eval_det_custom\detection_eval_custom.json" --output_dir "c:\Users\T2510600\Downloads\BAOCR\out\figs_compare"; python "c:\Users\T2510600\Downloads\BAOCR\evaluation\plot_recognition_results.py" --rec_eval "orig=c:\Users\T2510600\Downloads\BAOCR\out\merged_all\eval_rec_finetune\recognition_eval.json" "clahe=c:\Users\T2510600\Downloads\BAOCR\out\merged_all_clahe\eval_rec_finetune\recognition_eval.json" "highboost=c:\Users\T2510600\Downloads\BAOCR\out\merged_all_highboost\eval_rec_finetune\recognition_eval.json" --output_dir "c:\Users\T2510600\Downloads\BAOCR\out\figs_compare"
```

### 9) Extra detection visuals (optional)

- PR/F1 vs threshold (Original)
```powershell
foreach ($t in 0.20,0.30,0.40,0.50,0.60,0.70,0.80) { python "c:\Users\T2510600\Downloads\BAOCR\evaluation\evaluate_custom_db.py" --model "c:\Users\T2510600\Downloads\BAOCR\out\merged_all\runs\det_db_mnv3_custom_large_original\model_best.pt" --backbone large --test_data "c:\Users\T2510600\Downloads\BAOCR\out\merged_all\detection_test.json" --output_dir ("c:\Users\T2510600\Downloads\BAOCR\out\merged_all\thresh_{0}" -f ($t.ToString('0.00').Replace('.',''))) --mask_threshold $t }; python -c "import json,glob,os,matplotlib.pyplot as plt; base=r'c:\\Users\\T2510600\\Downloads\\BAOCR\\out\\merged_all'; pts=[]; [pts.append((int(os.path.basename(os.path.dirname(p)).split('_')[-1])/100.0,)+tuple(json.load(open(p,encoding='utf-8'))[k] for k in ['Precision@0.5','Recall@0.5','F1@0.5'])) for p in glob.glob(os.path.join(base,'thresh_*','detection_eval_custom.json'))]; pts=sorted(pts); ts,P,R,F1=zip(*pts); plt.figure(figsize=(6,4)); plt.plot(ts,P,'-o',label='Precision'); plt.plot(ts,R,'-o',label='Recall'); plt.plot(ts,F1,'-o',label='F1'); plt.xlabel('mask_threshold'); plt.ylabel('score'); plt.title('Original: PR/F1 vs threshold'); plt.legend(); plt.tight_layout(); os.makedirs(os.path.join(base,'figs'),exist_ok=True); out=os.path.join(base,'figs','pr_f1_vs_threshold_original.png'); plt.savefig(out,dpi=150); print('Saved',out)"
```

- IoU histograms and error overlays are produced under each `out/.../figs/` and `out/.../error_overlays/` folder using the one-liners from our session.

## Results Summary

### **Final Model Outputs**
- **Detection Models (3 variants):**
  - Original: `out/merged_all/eval_det_custom/`
  - CLAHE: `out/merged_all_clahe/eval_det_custom/`  
  - High-boost: `out/merged_all_highboost/eval_det_custom/`

- **Recognition Model (Final Combined):**
  - **Main Results**: `out/merged_all_combined/eval_rec_v3_reduced/`
  - **Training Logs**: `out/merged_all_combined/runs/rec_v3_reduced_combined_augplus/`
  - **Model Weights**: `model_best.pt` (62MB, optimized)



### **Visualization Outputs**
- **Training Curves**: `training_log.csv` (real-time monitoring)
- **Confusion Matrices**: Character-level error analysis
- **Sample Predictions**: Visual recognition results
- **Performance Plots**: CER/WER by language and length

## Monitoring & utilities

- **Tail detector training logs**
```powershell
Get-Content "c:\Users\T2510600\Downloads\BAOCR\out\merged_all\runs\det_db_mnv3_custom_large_original\training_log.csv" -Wait -Tail 40
```

- **Monitor final recognition training**
```powershell
Get-Content "c:\Users\T2510600\Downloads\BAOCR\out\merged_all_combined\runs\rec_v3_reduced_combined_augplus\training_log.csv" -Wait -Tail 40
```

- **Latest epoch every 5s (template)**
```powershell
while ($true) { $r = Import-Csv "<PATH_TO>/training_log.csv" | Select-Object -Last 1; if ($r) { "epoch=$($r.epoch) tr=$($r.train_loss) vl=$($r.val_loss) lr=$($r.lr)" }; Start-Sleep 5 }
```

- **List newest checkpoints (template)**
```powershell
Get-ChildItem "<RUN_DIR>" -Filter *.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 8 Name,Length,LastWriteTime
```

- **Stop a running training safely**
```powershell
# In the same window: Ctrl + C
Get-Process -Name python | Select-Object Id,ProcessName,StartTime,Path
Stop-Process -Id <PID> -Force
```

## Benchmarks table (auto-generate)

Generate a Markdown summary table combining detection and recognition metrics for Original/CLAHE/High-boost.

```powershell
python "c:\Users\T2510600\Downloads\BAOCR\evaluation\generate_benchmarks_md.py"
```

Output:

- `c:\Users\T2510600\Downloads\BAOCR\out\benchmarks\benchmarks.md`

