
# OCR Pipeline Flowchart

```
Input Image
     ↓
Preprocessing (CLAHE/High-boost/Original)
     ↓
Detection Model (MobileNetV3 + DB-like)
     ↓
Text Region Cropping
     ↓
Recognition Model (VGG16 + BiLSTM + Attention)
     ↓
CTC Decoding
     ↓
Final Text Output
```

## Model Specifications

**Detection Models (3 variants):**
- Architecture: Custom MobileNetV3 Large + DB-like segmentation
- Training: BCE+Dice loss, 100 epochs, patience=15
- Best: CLAHE variant (mAP@0.5 = 0.461)

**Recognition Model:**
- Architecture: VGG16_bn + Reduced BiLSTM + MultiheadAttention
- Features: 1 layer BiLSTM (128 units), 4-head attention (256 dim)
- Training: Combined dataset (59K samples), CTC loss + label smoothing
- Parameters: LR=2e-4, batch=12, 200 epochs, H=128 scaling
