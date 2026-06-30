# 🩺 Pneumonia Detection with Hybrid CNN-Transformer Ensemble

A deep learning project for pneumonia detection from chest X-ray images using a Hybrid CNN-Transformer architecture combined with Weighted Soft Voting Ensemble, K-Fold Cross Validation, and Explainable AI using Grad-CAM.

---

# 📌 Overview

This project develops a medical image classification system to detect **Pneumonia** from chest X-ray images using a hybrid deep learning architecture that combines:

* Convolutional Neural Network (CNN)
* Vision Transformer-inspired Encoder
* Ensemble Learning
* Explainable AI (Grad-CAM)

The pipeline includes:

* Data preprocessing and exploratory analysis
* Hybrid CNN-Transformer architecture design
* Hyperparameter tuning using Keras Tuner
* K-Fold Cross Validation
* Weighted Ensemble Learning
* Error analysis
* Threshold optimization
* Explainable AI visualization

Dataset used:
[Chest X-Ray Images - Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Target Classes:

* Normal
* Pneumonia

---

# 🔬 Methodology

The project is divided into 5 major stages.

---

# Step 1 — Data Analysis and Preprocessing

The dataset consists of chest X-ray grayscale images categorized into:

* NORMAL
* PNEUMONIA

## Dataset Distribution

| Dataset    | Normal | Pneumonia |
| ---------- | ------ | --------- |
| Train      | 1342   | 3876      |
| Validation | 9      | 9         |
| Test       | 234    | 390       |

Total Cross Validation Dataset:

* 5232 images

Total Test Dataset:

* 624 images

## Preprocessing Steps

Several preprocessing operations are applied:

* Resize image to `224 × 224`
* Convert images into grayscale
* Pixel normalization
* Dataset merging for cross-validation
* Statistical analysis of image intensity

## Exploratory Data Analysis (EDA)

EDA is performed to understand:

* Image distribution
* Class imbalance
* Pixel intensity characteristics
* Mean and standard deviation analysis

### Data Statistics

| Split      | Mean   | Std   |
| ---------- | ------ | ----- |
| Train      | 131.70 | 61.47 |
| Validation | 134.53 | 65.92 |
| Test       | 129.78 | 62.28 |

### Imbalance Ratio

Pneumonia : Normal = `0.35 : 1`

This imbalance motivates the use of:

* Class weighting
* Ensemble learning
* Threshold optimization

---

# Step 2 — Hybrid CNN-Transformer Architecture Design

The model combines CNN feature extraction with Transformer-based contextual learning.

## CNN Backbone

The CNN component contains:

* 3 Convolution Blocks
* Double Conv-BatchNorm layers
* MaxPooling
* Dropout Regularization

CNN is used to capture:

* Local texture patterns
* Lung opacity features
* Spatial abnormalities

## Transformer Encoder

Transformer blocks are added after CNN feature extraction using:

* Positional Encoding
* Multi-Head Attention
* GELU Activation
* L2 Regularization

Transformer helps the model learn:

* Global contextual relationships
* Long-range dependencies
* Attention-based representation

## Final Architecture

The final pipeline:

CNN Feature Extractor
→ Positional Encoding
→ Transformer Encoder
→ Classification Head

---

# Step 3 — Hyperparameter Tuning and Training

Hyperparameter optimization is performed using:

* Keras Tuner
* Validation Accuracy optimization

## Tuned Parameters

| Parameter            | Best Value |
| -------------------- | ---------- |
| CNN Filter Block 2   | 64         |
| CNN Filter Block 3   | 128        |
| CNN Dropout          | 0.1        |
| Projection Dimension | 64         |
| Transformer Layers   | 1          |
| Final Dropout        | 0.4        |
| Learning Rate        | 0.0001     |

## Optimization Strategy

The model uses:

* RMSprop Optimizer
* Binary Crossentropy
* Early Stopping
* ReduceLROnPlateau

Class weights are also applied to reduce imbalance effects.

---

# Step 4 — K-Fold Cross Validation and Ensemble Learning

To improve robustness and generalization capability, the project uses:

* 3-Fold Cross Validation
* Weighted Soft Voting Ensemble

## Ensemble Strategy

Each fold produces a trained model.

Predictions are combined using weighted averaging based on validation AUC scores.

### Validation AUC per Fold

| Fold   | Validation AUC |
| ------ | -------------- |
| Fold 1 | 0.9916         |
| Fold 2 | 0.9947         |
| Fold 3 | 0.9893         |

### Ensemble Weights

| Fold   | Weight |
| ------ | ------ |
| Fold 1 | 0.3332 |
| Fold 2 | 0.3343 |
| Fold 3 | 0.3325 |

## Performance Comparison

| Model             | Accuracy | AUC    |
| ----------------- | -------- | ------ |
| Single Best Model | 0.7676   | 0.9393 |
| Weighted Ensemble | 0.7933   | 0.9443 |

The ensemble model shows improved stability and predictive performance compared to a single model.

---

# Step 5 — Explainable AI and Error Analysis

To improve interpretability in medical prediction, Explainable AI methods are integrated into the pipeline.

## Grad-CAM Visualization

Grad-CAM is used to visualize important image regions influencing prediction decisions.

The visualization helps identify:

* Lung opacity regions
* Pneumonia-related abnormalities
* Attention focus of the model

## Error Analysis

Two major error types are analyzed:

| Error Type     | Count |
| -------------- | ----- |
| False Negative | 143   |
| False Positive | 2     |

### Interpretation

* The model is highly sensitive to pneumonia detection.
* Very few healthy patients are incorrectly diagnosed.
* However, some pneumonia cases are still missed.

---

# 📊 Threshold Optimization

Threshold analysis is performed to improve clinical usability.

## Recommended Threshold

| Threshold | Sensitivity | Specificity |
| --------- | ----------- | ----------- |
| 0.75      | 0.933       | 0.769       |

This threshold provides a better balance between:

* Detecting pneumonia cases
* Reducing false alarms

---

# 📈 Final Evaluation Results

## Classification Report

| Metric    | Normal | Pneumonia |
| --------- | ------ | --------- |
| Precision | 0.9785 | 0.7307    |
| Recall    | 0.3889 | 0.9949    |
| F1-Score  | 0.5566 | 0.8426    |

## Overall Metrics

| Metric      | Value  |
| ----------- | ------ |
| Accuracy    | 76.76% |
| AUC         | 0.9393 |
| Sensitivity | 0.9949 |
| Specificity | 0.3889 |

---

# 📌 Interpretation

The model achieves:

* Very high sensitivity for pneumonia detection
* Strong AUC performance
* Good ensemble stability

The high sensitivity indicates that the model is highly effective in identifying pneumonia patients, which is critical for medical screening systems.

However, specificity remains relatively low, indicating that the model tends to classify many normal images as pneumonia. This trade-off is common in healthcare applications where missing positive cases is considered more dangerous than false alarms.

The integration of Grad-CAM significantly improves model transparency and interpretability.

---

# 🛠️ Tech Stack

| Library            | Purpose                   |
| ------------------ | ------------------------- |
| TensorFlow / Keras | Deep learning framework   |
| NumPy              | Numerical computation     |
| Pandas             | Data manipulation         |
| OpenCV             | Image preprocessing       |
| Matplotlib         | Data visualization        |
| Seaborn            | Statistical visualization |
| Scikit-learn       | Evaluation metrics        |
| Keras Tuner        | Hyperparameter tuning     |

---

# 🧠 Model Components

Main components used in this project:

* CNN Backbone
* Transformer Encoder
* Positional Encoding
* Weighted Soft Voting Ensemble
* K-Fold Cross Validation
* Grad-CAM Explainability

---

# 📌 Conclusion

This project successfully develops a Hybrid CNN-Transformer ensemble model for pneumonia detection using chest X-ray images.

The integration of:

* CNN local feature extraction
* Transformer contextual learning
* Ensemble learning
* Explainable AI

Experimental results demonstrate that the model achieves:

* Excellent pneumonia sensitivity
* Strong AUC performance
* Good generalization capability

Several improvements can be implemented to further enhance the performance and robustness of this pneumonia detection model. First, future work can explore the use of larger and more diverse chest X-ray datasets from multiple medical institutions to improve model generalization and reduce dataset bias. Second, advanced architectures such as Vision Transformer (ViT), EfficientNet, or attention-based hybrid networks can be investigated to improve feature extraction and classification performance. Third, additional explainability methods such as SHAP, LIME, or integrated attention visualization can be incorporated to provide deeper interpretability and increase clinical trustworthiness for real-world medical applications.
