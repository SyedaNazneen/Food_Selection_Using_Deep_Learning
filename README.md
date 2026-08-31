# 🍔 Food Selection Using Deep Learning

An AI-powered Food Classification System built using **Deep Learning** and **Transfer Learning**.

The application allows users to upload a food image and classify it using three different Deep Learning architectures:

- 🧠 Custom CNN
- 🧠 VGG16
- 🧠 ResNet50

The system compares the performance of all models and provides the predicted food category along with prediction confidence and Top 3 predictions.

---

## 🚀 Live Application

🔗 Live Demo:https://foodselectionusingdeeplearning-16.streamlit.app/

---

## 📌 Project Overview

Food image classification is a Computer Vision problem where a Deep Learning model identifies the category of food present in an image.

This project implements and compares three different Deep Learning architectures:

| Model | Validation Accuracy |
|------|--------------------|
| Custom CNN | 64.35% |
| VGG16 | 75.12% |
| 🏆 ResNet50 | **82.94%** |

Based on validation accuracy, **ResNet50 achieved the best performance**.

---

# 🧠 Deep Learning Models

## 1️⃣ Custom CNN

A Convolutional Neural Network built from scratch for food image classification.

Features include:

- Convolution Layers
- Max Pooling
- Batch Normalization
- Dropout
- Fully Connected Layers

Validation Accuracy:

**64.35%**

---

## 2️⃣ VGG16

VGG16 is a Transfer Learning model that uses pre-trained convolutional layers for feature extraction.

Features:

- Pre-trained ImageNet weights
- Transfer Learning
- Fine-tuning
- Deep feature extraction

Validation Accuracy:

**75.12%**

---

## 3️⃣ ResNet50

ResNet50 is a deep residual neural network that uses skip connections to improve training performance.

Features:

- Residual Connections
- Deep Feature Extraction
- Transfer Learning
- Fine-tuning

Validation Accuracy:

**82.94%**

🏆 **Best Performing Model: ResNet50**

---

# 📊 Model Comparison

```text
Custom CNN  → 64.35%
VGG16       → 75.12%
ResNet50    → 82.94%
```

---

# 🍕 Supported Food Classes

The system supports **34 food categories**.

Examples include:

* Pizza
* Burger
* Butter Naan
* Donut
* Sandwich
* Sushi
* Taco
* Samosa
* Pav Bhaji
* Momos
* Paani Puri
* Jalebi
* Kulfi

And more.

---

# ✨ Application Features

* 📤 Upload Food Image
* 🧠 Select Deep Learning Architecture
* 🤖 Food Classification
* 🎯 Prediction Confidence
* 🏆 Top 3 Predictions
* 📊 Model Accuracy Comparison
* 📈 Accuracy Visualization
* 🕒 Prediction History
* 🏆 Best Model Information
* 👩‍💻 Professional Developer Profile Section

---

# 🖥️ Application Interface

The Streamlit application provides a modern and interactive interface with three main sections:

### 👤 User Information

Displays developer information and primary skill.

### 🤖 Food Prediction

Allows users to:

1. Upload a food image.
2. Select CNN, VGG16, or ResNet50.
3. Click the Predict Food button.
4. View the predicted food category.

### 📊 Model Comparison

Displays validation accuracy comparison between all models.

---

# 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* Streamlit
* NumPy
* Pandas
* Pillow
* Deep Learning
* Computer Vision
* Transfer Learning

---

# 📂 Project Structure

```text
Food_Selection_Deep_Learning_CNN/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── food_classification_cnn.keras
│   ├── food_classification_vgg16.keras
│   └── food_classification_resnet50.keras
│
├── json_data/
│   └── classes.json
│
├── metrics/
│   ├── cnn_accuracy_graph.png
│   ├── cnn_loss_graph.png
│   ├── vgg16_accuracy_graph.png
│   ├── vgg16_loss_graph.png
│   ├── resnet50_accuracy_graph.png
│   ├── resnet50_loss_graph.png
│   ├── model_comparison_summary.json
│   └── evaluation_summary.json
│
├── assets/
│   └── profile.png
│
├── test_images/
│
├── train_cnn_model.py
├── train_vgg16.py
├── train_resnet50.py
│
├── evaluate_vgg16_model.py
├── evaluate_resnet50_model.py
│
└── compare_models.py
```

---

# ⚙️ Installation

## Clone the Repository

```bash
git clone https://github.com/SyedaNazneen/Food_Selection_Using_Deep_Learning.git
```

## Navigate to Project Directory

```bash
cd Food_Selection_Using_Deep_Learning
```

## Create Virtual Environment

```bash
python -m venv .venv
```

## Activate Virtual Environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

The application will start locally.

---

# 📈 Model Performance

The trained models were evaluated using validation data.

### Final Results

| Model       | Accuracy   |
| ----------- | ---------- |
| Custom CNN  | 64.35%     |
| VGG16       | 75.12%     |
| 🏆 ResNet50 | **82.94%** |

ResNet50 achieved the highest validation accuracy and was selected as the best-performing model.

---

# 🔮 Future Improvements

Possible future enhancements include:

* Increase dataset size
* Add more food categories
* Improve model accuracy
* Implement ensemble learning
* Add Grad-CAM visualization
* Deploy using Streamlit Cloud
* Add API support
* Create a mobile application

---

# 👩‍💻 Developer

**Syeda Nazneen**

Deep Learning Developer | Python | Machine Learning | Computer Vision

# GitHub:

https://github.com/SyedaNazneen

---

# ⭐ Support

If you like this project, please consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is created for educational and portfolio purposes.


