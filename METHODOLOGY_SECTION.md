# 1.5 METHODOLOGY

## 1.5.1 Overview

This research presents a comprehensive Bengali-English OCR pipeline combining text detection and recognition models with preprocessing optimization. The methodology follows a systematic approach: dataset preparation, preprocessing enhancement, model architecture design, training optimization, and performance evaluation.

## 1.5.2 Dataset Preparation

### 1.5.2.1 Data Collection and Annotation
The dataset comprises 450 handwritten Bengali-English document images collected from five sources:
- **d1**: 85 images with Label Studio annotations (`d1_annotations.json`)
- **d2**: 47 images with Label Studio annotations (`d2_annotations.json`)
- **d3**: 20 images with Label Studio annotations (`d3_annotations.json`)
- **d4**: 38 images with Label Studio annotations (`d4_annotations.json`)
- **raw**: 260 images with individual per-image JSON annotations

**📊 DIAGRAM AVAILABLE**: `sample_images/complete_pipeline_example.png` shows the complete workflow

### 1.5.2.2 Data Conversion and Splitting
Label Studio annotations were converted to DocTR-compatible format for both detection and recognition tasks. The combined dataset was split using stratified sampling:
- **Training**: 70% (315 images)
- **Validation**: 15% (68 images)
- **Test**: 15% (67 images)

**📊 DIAGRAM NEEDED**: Dataset distribution pie chart showing source breakdown and train/val/test splits

## 1.5.3 Image Preprocessing

### 1.5.3.1 Preprocessing Variants
Three preprocessing approaches were implemented to enhance text visibility:

1. **Original**: Baseline images without modification
2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: 
   - Clip limit: 2.0
   - Tile grid size: 8×8
   - Enhances local contrast while preventing over-amplification
3. **High-boost Filtering**: 
   - Kernel: [-1, -1, -1; -1, 9, -1; -1, -1, -1]
   - Emphasizes edges and fine details in handwritten text

**📊 DIAGRAM AVAILABLE**: `sample_images/detection_comparison_sample_1.png` shows visual comparison of preprocessing effects

### 1.5.3.2 Dataset Expansion Strategy
To maximize training data, all preprocessing variants were combined:
- **Original dataset**: 5,663 recognition samples
- **CLAHE dataset**: 27,897 samples  
- **High-boost dataset**: 25,897 samples
- **Combined dataset**: 59,457 total samples (10.5× expansion)

**📊 DIAGRAM NEEDED**: Bar chart showing dataset size progression from original to combined

## 1.5.4 Detection Model Architecture

### 1.5.4.1 Model Design
Custom MobileNetV3-based detection architecture with DB-like segmentation:
- **Backbone**: MobileNetV3 Large (pretrained on ImageNet)
- **Decoder**: Lightweight upsampling layers
- **Output**: Binary segmentation masks converted to bounding boxes
- **Loss Function**: Combined BCE + Dice loss for robust segmentation

### 1.5.4.2 Training Configuration
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-4)
- **Batch Size**: 4 (memory constraints)
- **Epochs**: 100 with early stopping (patience=15)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)

**📊 DIAGRAM AVAILABLE**: Architecture shown in `sample_images/complete_pipeline_example.png`
**📊 DIAGRAM NEEDED**: Detailed MobileNetV3 detection architecture diagram

## 1.5.5 Recognition Model Architecture

### 1.5.5.1 CRNN with Attention Enhancement
The recognition model implements a reduced-complexity CRNN with attention mechanism:

```
Input (H=128, W=variable) 
    ↓
VGG16_bn Backbone (pretrained)
    ↓
Feature Maps (256 channels)
    ↓
Bidirectional LSTM (1 layer, 128 hidden units)
    ↓
MultiheadAttention (4 heads, 256 embed_dim)
    ↓
Linear Layer (vocab_size outputs)
    ↓
CTC Loss
```

### 1.5.5.2 Model Optimizations
Following faculty recommendations:
- **Image scaling**: Increased from H=32 to H=128 (4× resolution)
- **Reduced complexity**: 1 BiLSTM layer (vs. 2), 128 hidden units (vs. 256)
- **Attention integration**: 4-head MultiheadAttention for sequence modeling
- **Label smoothing**: 0.1 for CTC loss regularization

**📊 DIAGRAM NEEDED**: Detailed CRNN+Attention architecture flowchart

### 1.5.5.3 Training Configuration
- **Optimizer**: Adam (lr=2e-4, weight_decay=1e-4, amsgrad=True)
- **Batch Size**: 12 (optimized for combined dataset)
- **Epochs**: 200 with early stopping (patience=25)
- **Gradient Clipping**: max_norm=5.0
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=8)

**📊 DIAGRAM AVAILABLE**: Training curves in `curvesfigures/recognition_training_curves.png`

## 1.5.6 Data Augmentation Strategy

### 1.5.6.1 Handwriting-Specific Augmentations
Tailored augmentation pipeline for Bengali handwritten text:
- **Geometric**: Rotation (±7°), shift-scale-rotate, elastic transform, grid distortion
- **Photometric**: Random brightness/contrast, Gaussian noise, blur variants
- **Sharpening**: Alpha=(0.2, 0.5) for text clarity enhancement
- **Removed**: CoarseDropout (counterproductive for character learning)

### 1.5.6.2 Augmentation Rationale
- **Conservative rotation**: ±7° preserves Bengali character structure
- **Elastic transforms**: Simulates natural handwriting variations
- **No destructive augmentation**: Maintains character integrity during learning phase

**📊 DIAGRAM NEEDED**: Before/after augmentation examples showing Bengali text transformations

## 1.5.7 Training Optimization

### 1.5.7.1 Smart Checkpoint Management
Implemented efficient storage strategy:
- **Keep**: Best model + 3 most recent checkpoints
- **Delete**: Older checkpoints automatically
- **Storage reduction**: 98% space savings vs. keeping all checkpoints

### 1.5.7.2 Multi-Model Training Strategy
- **Detection**: 3 separate models for Original/CLAHE/High-boost variants
- **Recognition**: Single model trained on combined dataset for generalization
- **Evaluation**: Cross-validation across all preprocessing combinations

**📊 DIAGRAM AVAILABLE**: Performance comparison in `curvesfigures/detection_metrics_comparison.png`

## 1.5.8 Evaluation Methodology

### 1.5.8.1 Detection Metrics
- **Precision/Recall/F1**: At IoU thresholds 0.5 and 0.75
- **mAP**: Mean Average Precision across IoU range
- **Average IoU**: Spatial accuracy of detected regions

### 1.5.8.2 Recognition Metrics
- **Character Error Rate (CER)**: Character-level accuracy
- **Word Error Rate (WER)**: Word-level accuracy  
- **Language-specific**: Separate Bengali and English evaluation
- **Confidence Analysis**: Model certainty assessment

**📊 DIAGRAM AVAILABLE**: 
- Language comparison: `curvesfigures/recognition_language_comparison.png`
- Sample results: `sample_images/recognition_samples_grid_real.png`

### 1.5.8.3 Error Analysis
Systematic analysis of failure modes:
- **Character confusions**: Bengali-specific OCR challenges (ন/ণ, ব/ভ)
- **Confidence correlation**: Relationship between model certainty and accuracy
- **Length dependency**: Performance variation with text length

**📊 DIAGRAM AVAILABLE**: Confidence distribution in `curvesfigures/recognition_confidence_histogram.png`

## 1.5.9 Implementation Framework

### 1.5.9.1 Software Stack
- **Deep Learning**: PyTorch 1.12+ with CUDA support
- **Computer Vision**: OpenCV, PIL for image processing
- **Data Augmentation**: Albumentations library
- **Evaluation**: Custom metrics implementation with DocTR compatibility

### 1.5.9.2 Hardware Configuration
- **GPU**: CUDA-enabled for model training and inference
- **Memory Management**: Optimized batch sizes and gradient accumulation
- **Storage**: Efficient checkpoint management for large-scale training

**📊 DIAGRAM NEEDED**: System architecture diagram showing software/hardware components

---

## REQUIRED DIAGRAMS SUMMARY

### ✅ **AVAILABLE in sample_images/ and curvesfigures/**:
1. Complete pipeline workflow (`sample_images/complete_pipeline_example.png`)
2. Preprocessing comparison (`sample_images/detection_comparison_sample_*.png`)
3. Training curves (`curvesfigures/recognition_training_curves.png`)
4. Performance metrics (`curvesfigures/detection_metrics_comparison.png`)
5. Language comparison (`curvesfigures/recognition_language_comparison.png`)
6. Sample results (`sample_images/recognition_samples_grid_real.png`)
7. Confidence analysis (`curvesfigures/recognition_confidence_histogram.png`)

### 📋 **NEEDED (to be created)**:
1. **Dataset distribution diagram**: Pie/bar charts showing source breakdown and splits
2. **Dataset expansion chart**: Bar chart showing 5,663 → 59,457 sample growth
3. **MobileNetV3 detection architecture**: Detailed network diagram
4. **CRNN+Attention architecture**: Detailed recognition model flowchart
5. **Data augmentation examples**: Before/after Bengali text transformations
6. **System architecture**: Software/hardware component diagram

### 💡 **Creation Suggestions**:
- Use **draw.io** or **Lucidchart** for architecture diagrams
- Use **matplotlib/seaborn** for statistical charts
- Use **PowerPoint** for simple flowcharts and system diagrams
- Maintain consistent color scheme with existing visualizations
