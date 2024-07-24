# Hepatocyte Ballooning Detection in Liver Ultrasound Images: Phase 2 Documentation

## Project Overview

This project continues the development of a deep learning system for automatically detecting hepatocyte ballooning in liver ultrasound images. The goal remains to assist in the diagnosis of non-alcoholic fatty liver disease (NAFLD) using AI-driven image analysis.

### Key Objectives
- Implement and evaluate deep learning models for multi-class classification of hepatocyte ballooning
- Address significant class imbalance in the dataset
- Utilize transfer learning with medical imaging-specific pre-trained models
- Implement robust cross-validation strategies
- Compare optimizer performance (Adam vs AdamW)
- Prepare comprehensive results for potential publication

## Dataset

### Classes
- None (0): Normal liver cells
- Few Balloon Cells (1): Minor hepatocyte ballooning
- Many Cells/Prominent Ballooning (2): Significant hepatocyte ballooning

### Class Distribution
- None: 2236 images (1330 new + 906 original)
- Few: 6232 images (492 new + 5740 original)
- Many: 256 images (151 new + 105 original)

### Class Imbalance
Despite the addition of new images, the dataset still exhibits significant class imbalance, particularly for the 'Many' class. This continues to pose a challenge for model training and evaluation.

## Methodology

### Data Preprocessing and Augmentation

#### Two-Phase Augmentation Strategy (Maintained from Phase 1)
1. First Phase (Saved Augmentations):
   - Random rotations (limited to 15 degrees)
   - Horizontal flips
   - Random shifts and scaling
   - Intensity adjustments
   - Elastic transformations (for 'Many' class)

2. Second Phase (During Training):
   - Subtle shifts and intensity changes
   - Gaussian noise

#### Implementation
- Continued use of Albumentations library for efficient image augmentations
- Maintained custom Dataset and DataLoader classes for on-the-fly augmentation

### Model Development

#### Deep Learning Models Implemented
1. InceptionV3 (pre-trained)
2. EfficientNetB0 (trained from scratch)

#### Transfer Learning
- Utilized pre-trained weights from RadImageNet for InceptionV3

### Training Strategy

- Implemented class weighting to address imbalance
- Used custom dataset class for efficient, on-the-fly augmentation
- Employed balanced batch sampling
- Implemented 10-fold cross-validation for robust performance estimation

### Optimizer Comparison
- Compared performance of Adam and AdamW optimizers

### Evaluation Metrics

- Accuracy
- Sensitivity (per class and average)
- Specificity (per class and average)
- ROC-AUC
- F1-score (per class and average)
- Loss (train and validation)

## Results Summary

### Best Performing Model: InceptionV3
- Best Validation Accuracy: 87.51% (Fold 4, Epoch 17)
- Best AUC: 0.9322 (Fold 1, Epoch 28)

### Model Performance Overview (10-fold Cross-Validation)

| Model          | Avg Val Accuracy | Avg AUC | Avg F1-score |
|----------------|-------------------|---------|--------------|
| InceptionV3    | 85.86%            | 0.9163  | 0.8609       |
| EfficientNetB0 | 64.86%            | 0.7493  | 0.6545       |

### Class-wise Performance (InceptionV3, Averaged across folds)

| Class | Sensitivity | Specificity | F1-score |
|-------|-------------|-------------|----------|
| None  | 77.26%      | 89.70%      | 0.8285   |
| Few   | 88.75%      | 78.53%      | 0.8310   |
| Many  | 69.23%      | 98.67%      | 0.8056   |

### Optimizer Comparison (EfficientNetB0)

| Optimizer | Avg Val Accuracy | Avg AUC | Avg F1-score |
|-----------|-------------------|---------|--------------|
| Adam      | 64.86%            | 0.7493  | 0.6545       |
| AdamW     | 63.09%            | 0.7577  | 0.6467       |

## Challenges and Observations

1. Persistent class imbalance, particularly for the 'Many' class, impacting model performance
2. Improved stability in performance on the 'Many' class compared to Phase 1
3. Significant performance gap between InceptionV3 and EfficientNetB0, highlighting the benefits of transfer learning
4. Minimal difference in performance between Adam and AdamW optimizers
5. 10-fold cross-validation provided more robust performance estimates

## Technical Implementation

- Framework: PyTorch
- Primary Libraries: torchvision, Albumentations
- Hardware: NVIDIA RTX 3060 Ti GPU

## Conclusions and Next Steps

1. InceptionV3 with transfer learning from RadImageNet significantly outperforms EfficientNetB0 trained from scratch
2. 10-fold cross-validation provided more reliable performance estimates
3. Class imbalance remains a challenge, but augmentation and class weighting strategies show improvement
4. Adam and AdamW optimizers perform similarly for this task
5. Further steps:
   - Explore ensemble methods to leverage strengths of different models
   - Investigate decision boundary optimization for clinical relevance
   - Consider implementing more advanced augmentation techniques
   - Explore other transfer learning sources specific to ultrasound imaging
   - Prepare comprehensive documentation and analysis for publication

This phase has provided valuable insights into model performance and optimization strategies for hepatocyte ballooning detection in liver ultrasound images. The results demonstrate the potential of deep learning in assisting NAFLD diagnosis, while also highlighting areas for further improvement and research.