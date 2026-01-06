# WIDS_5.0-End-to-End-ML
# Plant Disease Classification using Machine Learning

This repository contains exploratory data analysis (EDA), baseline machine learning models, and deep Learning models - Simple CNN and Transfer Learning built on the **PlantVillage dataset**.  
The work was done as part of **WiDS 5.0 (Week 1, Week 2, and Week 3)**

---

## Objective

- Understand the structure and challenges of the PlantVillage image dataset  
- Perform exploratory data analysis on image quality and distribution  
- Build simple baseline machine learning models  
- Handle class imbalance and evaluate its effect on performance  
- Compare the performance of Simple CNN model and model built using Transfer Learning
---

## Dataset Description

- Total images: **54,305**
- Number of plant species: **14**
- Number of classes: **38**
- Problem setup: **Binary Classification** (initial)
  - `0` → Healthy
  - `1` → Diseased

Images are collected under two conditions:
1. Healthy plant leaves  
2. Diseased plant leaves  

---

## Exploratory Data Analysis (EDA)

### Image Properties
- Resolution: **256 × 256**
- Aspect Ratio: **1:1**
- All images have the same size and shape

### Class Distribution
- Dataset is **highly imbalanced**
- Some classes have more than **5000 images**
- Some classes have fewer than **500 images**
- This imbalance can bias models towards dominant classes

### Blur Analysis
- Many images have **low blur scores**
- A significant portion of the dataset contains blurry images
- Blurry images may negatively affect model performance

### Brightness Analysis
- Most images have brightness values around the mean
- Very dark images are rare
- Overall lighting is fairly consistent

### Background Analysis
- Majority of images have **light backgrounds**
- Around **2.5%** images have black backgrounds
- Some images contain only the leaf with no visible background
- Background variation may act as a shortcut feature for models

---

## Feature Engineering

Steps used to create the feature matrix:

1. Resize each image to **64 × 64**
2. Each image becomes an array of shape `(64, 64, 3)`
3. Flatten into a vector of size `12288`
4. Stack all images into a matrix of shape `(54305, 12288)`

### Label Distribution
- Diseased (1): ~72%
- Healthy (0): ~28%

---

## Baseline Models

### Dummy Baseline
- Always predicts the majority class (`Diseased`)
- Accuracy: **~72%**
- Any useful model must perform better than this

---

### Shallow Machine Learning Models

The following models were trained:

- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest  
- XGBoost  

#### Results Summary
- Logistic Regression: ~83% accuracy  
- Random Forest: ~89% accuracy (biased towards diseased class)  
- XGBoost: ~92% accuracy (best overall performance)  

XGBoost performed the best despite class imbalance.

---

## Handling Class Imbalance

- Used **SMOTE (Synthetic Minority Oversampling Technique)**
- Minority class was oversampled to match the majority class
- Models retrained on balanced data

### After SMOTE
- Logistic Regression accuracy improved to ~89%
- Random Forest accuracy improved to ~94%
- Precision and recall became more balanced

---

## Key Observations

- Class imbalance is a major issue in this dataset
- Image quality (blur and background) affects learning
- Classical ML models can achieve strong performance
- Proper preprocessing is very important before using deep models

---

## Simple CNN Model

After establishing classical machine learning baselines, a **simple Convolutional Neural Network (CNN)** was trained to capture spatial features directly from images.

### Model Overview
- Input: RGB images
- Task: Classification- 38 classes
- Loss function: Categorical / Binary Cross-Entropy
- Optimizer: Adam
- Epochs: 10
- Training setup:
  - Train set
  - Validation set
  - Separate test set for final evaluation

The CNN was intentionally kept shallow to act as a **deep learning baseline**, not a highly optimized architecture.

---

## Training Performance

During training, the model showed rapid convergence:

- Training accuracy increased steadily and reached **~99%**
- Validation accuracy stabilized around **85–87%**
- Validation loss stopped improving after a few epochs, indicating mild overfitting

This behavior is expected due to:
- Class imbalance
- Limited regularization
- Simple architecture

---

## Test Set Results

Final evaluation on the unseen test dataset:

- **Test Accuracy:** `~86.7%`
- **Test Loss:** `~0.897`

These results indicate that the CNN generalizes reasonably well and performs competitively with classical ML baselines, while learning directly from image pixels.

---

## Observations

- CNN outperforms Logistic Regression but is slightly below XGBoost
- Strong training accuracy suggests sufficient model capacity
- Gap between training and validation accuracy indicates:
  - Overfitting
  - Need for better regularization or data augmentation

---

## Transfer Learning using EfficientNetB0

To further improve performance, **transfer learning** was used with a pre-trained **EfficientNetB0** model.  
EfficientNetB0 is trained on ImageNet and is able to extract strong, general-purpose visual features.

Only the final classification layers were trained, while the base model was kept frozen.

---

## Model Details

- Base model: **EfficientNetB0 (ImageNet weights)**
- Input: RGB images
- Task: Classification- 38 classes
- Training strategy:
  - Feature extraction using pre-trained backbone
  - Custom classification head on top
- Optimizer: Adam
- Loss function: Binary / Categorical Cross-Entropy
- Epochs: **5**

---

## Training Performance

The model converged very quickly due to transfer learning:

- Training accuracy increased rapidly and reached **~98%**
- Validation accuracy remained stable around **84–87%**
- Fewer epochs were required compared to training a CNN from scratch

This shows the advantage of using pre-trained representations.

---

## Test Set Results

Final evaluation on the unseen test dataset:

- **Test Accuracy:** `~84.8%`
- **Test Loss:** `~0.84`

---

## Observations

- Transfer learning achieved competitive performance with fewer epochs
- Model trains faster compared to simple CNN
- Slight drop in test accuracy compared to CNN suggests:
  - Possible domain mismatch
  - Need for fine-tuning deeper layers

---

## Comparison with Other Models

| Model | Test Accuracy |
|------|--------------|
| Logistic Regression | ~83% |
| Random Forest | ~89% |
| XGBoost | ~92% |
| Simple CNN | ~86.7% |
| EfficientNetB0 (Transfer Learning) | ~84.8% |

---

## Possible Improvements

- Fine-tune upper layers of EfficientNetB0
- Apply stronger data augmentation
- Use class weights or focal loss
- Increase training epochs with early stopping

