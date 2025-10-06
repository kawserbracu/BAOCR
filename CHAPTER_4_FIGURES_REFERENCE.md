# CHAPTER 4 FIGURES REFERENCE GUIDE

## 📊 AVAILABLE FIGURES (Ready to Use)

### From `sample_images/` folder:
1. **`complete_pipeline_example.png`** 
   - **Section**: 4.1.1, 4.4.2
   - **Caption**: "Complete Bengali-English OCR pipeline workflow showing the integration of detection and recognition components with technical specifications."

2. **`detection_comparison_sample_1.png`** (and sample_2, sample_3)
   - **Section**: 4.3.2, 4.3.4
   - **Caption**: "Visual comparison of preprocessing effects on detection performance. CLAHE preprocessing (green boxes) demonstrates superior boundary detection compared to Original (red) and High-boost (blue) variants."

3. **`recognition_samples_grid_real.png`**
   - **Section**: 4.8.4
   - **Caption**: "Real Bengali OCR recognition samples showing ground truth vs. predictions. Top rows show correct predictions with high confidence, bottom rows demonstrate typical error patterns including character confusions."

### From `curvesfigures/` folder:
4. **`detection_metrics_comparison.png`**
   - **Section**: 4.3.4, 4.8.1
   - **Caption**: "Comprehensive detection performance comparison across preprocessing variants showing mAP, Precision, Recall, F1-score, and IoU metrics."

5. **`recognition_training_curves.png`**
   - **Section**: 4.5.5, 4.7.3
   - **Caption**: "Training and validation curves for the recognition model showing loss and accuracy progression over epochs with early stopping behavior."

6. **`recognition_language_comparison.png`**
   - **Section**: 4.8.2
   - **Caption**: "Language-specific performance comparison showing Character Error Rate (CER) and Word Error Rate (WER) for Bengali and English text recognition."

7. **`recognition_confidence_histogram.png`**
   - **Section**: 4.8.3
   - **Caption**: "Distribution of prediction confidence scores showing model certainty patterns with average confidence threshold indicated."

## 📋 FIGURES NEEDED (To Be Created)

### 1. Dataset Distribution Diagram
- **Section**: 4.2.2
- **Type**: Pie chart + Bar chart
- **Content**: 
  - Pie chart: d1 (85), d2 (47), d3 (20), d4 (38), raw (260)
  - Bar chart: Train (315), Val (68), Test (67)
- **Tools**: matplotlib/seaborn or PowerPoint
- **Caption**: "Dataset composition showing source distribution (left) and train/validation/test split (right) with sample counts."

### 2. Dataset Expansion Progression
- **Section**: 4.3.3
- **Type**: Bar chart with growth indicators
- **Content**: Original (5,663) → CLAHE (27,897) → High-boost (25,897) → Combined (59,457)
- **Tools**: matplotlib with expansion multipliers (1×, 4.9×, 4.6×, 10.5×)
- **Caption**: "Dataset expansion strategy showing progressive sample count growth through preprocessing variants, achieving 10.5× total expansion."

### 3. MobileNetV3 Detection Architecture
- **Section**: 4.4.2
- **Type**: Technical architecture diagram
- **Content**: 
  - Input → MobileNetV3 Backbone → Decoder → Segmentation Mask → Bounding Boxes
  - Layer specifications and channel dimensions
- **Tools**: draw.io, Lucidchart, or PowerPoint
- **Caption**: "MobileNetV3-based detection architecture showing feature extraction backbone, decoder network, and post-processing pipeline for text region detection."

### 4. CRNN+Attention Architecture
- **Section**: 4.5.2
- **Type**: Detailed model flowchart
- **Content**:
  - VGG16_bn → BiLSTM → MultiheadAttention → Linear → CTC
  - Dimensions: (H=128, W) → (W/4, 512) → (W/4, 256) → (W/4, vocab_size)
- **Tools**: draw.io, Lucidchart, or PowerPoint
- **Caption**: "Recognition model architecture combining VGG16 backbone, bidirectional LSTM, multi-head attention mechanism, and CTC loss for sequence-to-sequence learning."

### 5. Data Augmentation Examples
- **Section**: 4.6.2
- **Type**: Before/after image grid
- **Content**: 
  - Original Bengali text samples
  - After rotation, elastic transform, brightness/contrast, noise
- **Tools**: Python script with actual samples or image editing
- **Caption**: "Data augmentation examples showing geometric and photometric transformations applied to Bengali handwritten text while preserving character integrity."

### 6. Preprocessing Effectiveness Comparison
- **Section**: 4.3.4
- **Type**: Side-by-side performance bars
- **Content**: 
  - mAP@0.5: Original (0.431), CLAHE (0.461), High-boost (0.442)
  - F1@0.5: Original (0.777), CLAHE (0.793), High-boost (0.771)
- **Tools**: matplotlib with highlighted best performer
- **Caption**: "Preprocessing effectiveness comparison showing CLAHE's superior performance in both mAP and F1 scores for text detection tasks."

## 🛠️ FIGURE CREATION TOOLS AND SUGGESTIONS

### For Statistical Charts (Figures 1, 2, 6):
```python
# Use matplotlib/seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset distribution pie chart
plt.pie(counts, labels=sources, autopct='%1.1f%%')
plt.title('Dataset Source Distribution')

# Bar charts with value annotations
bars = plt.bar(categories, values)
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
             str(value), ha='center', va='bottom')
```

### For Architecture Diagrams (Figures 3, 4):
- **draw.io** (free, web-based): Best for technical diagrams
- **Lucidchart**: Professional diagramming tool
- **PowerPoint**: Simple flowcharts and block diagrams
- **Visio**: Advanced technical diagrams

### For Image Processing Examples (Figure 5):
```python
# Create augmentation examples
import albumentations as A
import cv2

# Apply augmentations to real Bengali samples
transform = A.Compose([
    A.Rotate(limit=7, p=1.0),
    A.ElasticTransform(alpha=1.5, sigma=25, p=1.0),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0)
])

# Show before/after grid
```

## 📏 FIGURE SPECIFICATIONS

### Technical Requirements:
- **Resolution**: 300 DPI minimum for thesis quality
- **Format**: PNG or PDF for vector graphics
- **Size**: Fit within thesis page margins (typically 6-7 inches width)
- **Font**: Consistent with thesis font (Times New Roman or similar)
- **Colors**: High contrast, colorblind-friendly palette

### Placement Guidelines:
- **Position**: Near first reference in text
- **Numbering**: Figure 4.1, 4.2, etc. in order of appearance
- **Captions**: Below figures, descriptive and self-contained
- **References**: Proper in-text citations "as shown in Figure 4.X"

## 🎯 PRIORITY ORDER FOR CREATION

### High Priority (Essential):
1. **Dataset Distribution** - Shows research scope
2. **CRNN+Attention Architecture** - Core technical contribution
3. **Dataset Expansion** - Methodology innovation

### Medium Priority (Important):
4. **MobileNetV3 Architecture** - Technical completeness
5. **Preprocessing Effectiveness** - Validates approach

### Lower Priority (Nice to have):
6. **Data Augmentation Examples** - Implementation details

This prioritization helps you focus on the most critical figures first while ensuring comprehensive documentation of your methodology.
