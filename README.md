# 📑 Analyzing the Effectiveness of Convolution Neural Networks in Automated Skin Cancer Classification

## 📌 Overview
This research investigates the performance of **ResNet50** and **VGG16** convolutional neural networks (CNNs) in classifying skin lesion images for **automated diagnosis of skin cancer**. The study evaluates their effectiveness based on **accuracy, F1-score, and confusion matrices**, using the **HAM10000** dataset.

## 📚 Table of Contents
- [Introduction](#introduction)
- [Background](#background)
- [Methodology](#methodology)
- [Results](#results)
- [Limitations](#limitations)
- [Conclusion](#conclusion)
- [Code & Data](#code--data)
- [References](#references)

---

## 🧑‍⚕️ Introduction
With the **rise of AI in healthcare**, CNNs have shown promise in **medical image analysis**. This study compares two CNN architectures, **VGG16** and **ResNet50**, to determine their effectiveness in skin cancer detection.

Key questions:
- Which model achieves **higher accuracy** in classifying skin lesion images?
- How do different **architectures impact generalization and overfitting**?

---

## 🤖 Background
### 🔹 Neural Networks & CNNs
- **Artificial Neural Networks (ANNs)** mimic the human brain’s ability to learn from examples.
- **CNNs**, a specialized form of ANNs, excel at **image classification tasks** through layers of **convolution, pooling, and fully connected layers**.
- CNNs are widely used in **medical applications**, including skin cancer classification.

### 🔹 VGG16 vs. ResNet50
- **VGG16**: A classic **16-layer** CNN with a simple architecture.
- **ResNet50**: A deeper **50-layer** network with **skip connections** to improve gradient flow and reduce overfitting.

---

## 🧪 Methodology
### 📊 Dataset: **HAM10000**
- **11,721 images** of skin lesions from **8 different classes**.
- Images labeled for **various skin cancer types**.

### 🏗️ Model Implementation
- **Pre-trained VGG16 and ResNet50 models** were fine-tuned.
- **Softmax activation function** was used in the output layer for classification.
- Training performed in **Google Colab** using the **Tensor T4 GPU**.

### 🔍 Evaluation Metrics
- **Accuracy**: Measures the proportion of correct predictions.
- **F1-Score**: Balances **precision** and **recall** for imbalanced datasets.
- **Confusion Matrix**: Provides insight into model misclassifications.

---

## 📊 Results
### ✅ Accuracy
| Model   | Training Accuracy | Testing Accuracy |
|---------|------------------|-----------------|
| ResNet50 | **82.0%** | **74.83%** |
| VGG16 | **79.0%** | **72.7%** |

### 📈 F1 Scores
| Model   | F1 Score |
|---------|---------|
| ResNet50 | **0.7158** |
| VGG16 | **0.7119** |

- **ResNet50 outperformed VGG16 in accuracy and F1-score.**
- **ResNet50 exhibited potential overfitting**, performing better on training data than test data.
- **VGG16 showed better distribution of correct predictions across all classes.**

### 🔄 Confusion Matrix Insights
- **ResNet50 overclassified "nevus" lesions**, suggesting a bias in its predictions.
- **VGG16 struggled with squamous cell carcinoma and dermatofibroma** due to dataset imbalance.

---

## 🚧 Limitations
1. **Dataset Imbalance**: Some classes had significantly fewer samples, affecting performance.
2. **Limited Training Epochs**: Models were trained for **only 30 epochs** due to runtime constraints.
3. **Pre-trained Models**: Custom implementations might yield different results.

---

## 🏁 Conclusion
- **ResNet50** achieved **higher accuracy** but showed signs of **overfitting**.
- **VGG16**, though slightly less accurate, demonstrated **better generalization**.
- Further research with **balanced datasets and extended training** could refine the results.

---

## 📂 Code & Data
- The **dataset (HAM10000)** is publicly available.
- Code for preprocessing, training, and evaluation is in **Appendix C** of the research paper.

---

## 🔗 References
- Tschandl, P., et al. (2018). "HAM10000 dataset."
- Esteva, A., et al. (2017). "Dermatologist-level classification of skin cancer with deep neural networks."
- IBM (n.d.). "Computer vision in AI applications."
