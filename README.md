# 🍔 Food Selection Using Deep Learning

<p align="center">
  <img src="assets/food_app_preview.png" alt="Food Selection Using Deep Learning Application Preview" width="100%">
</p>

<p align="center">
  <b>An AI-Powered Food Classification System using Deep Learning and Transfer Learning</b>
</p>

<p align="center">
  <a href="https://foodselectionusingdeeplearning-16.streamlit.app/">
    🚀 Live Demo
  </a>
  &nbsp; | &nbsp;
  <a href="https://github.com/SyedaNazneen/Food_Selection_Using_Deep_Learning">
    💻 GitHub Repository
  </a>
</p>

---

## 🚀 Live Application

Experience the deployed application here:

👉 **Live Demo:**  
https://foodselectionusingdeeplearning-16.streamlit.app/

The application allows users to upload a food image and classify it using Deep Learning models.

---

# 📌 Project Overview

Food image classification is a **Computer Vision and Deep Learning problem** where an AI model analyzes an uploaded food image and predicts its food category.

This project implements and compares **three Deep Learning architectures**:

- 🧠 Custom CNN
- 🧠 VGG16
- 🧠 ResNet50

The application provides:

- Predicted Food Category
- Prediction Confidence
- Top 3 Predictions
- Model Accuracy Comparison
- Best Performing Model Information

---

# 🧠 Deep Learning Models

## 1️⃣ Custom CNN

A Convolutional Neural Network built from scratch for food image classification.

### Features

- Convolution Layers
- Max Pooling
- Batch Normalization
- Dropout
- Fully Connected Layers

### Validation Accuracy

**64.35%**

---

## 2️⃣ VGG16

VGG16 is a popular Transfer Learning architecture that uses pre-trained convolutional layers for feature extraction.

### Features

- Pre-trained ImageNet Weights
- Transfer Learning
- Fine-Tuning
- Deep Feature Extraction

### Validation Accuracy

**75.12%**

---

## 3️⃣ ResNet50

ResNet50 is a powerful Deep Residual Neural Network that uses skip connections to improve training performance.

### Features

- Residual Connections
- Deep Feature Extraction
- Transfer Learning
- Fine-Tuning

### Validation Accuracy

🏆 **82.94%**

### 🥇 Best Performing Model

**ResNet50**

---

# 📊 Model Performance Comparison

| Model | Validation Accuracy |
|------|--------------------|
| Custom CNN | 64.35% |
| VGG16 | 75.12% |
| 🏆 ResNet50 | **82.94%** |

```text
Custom CNN  → 64.35%
VGG16       → 75.12%
ResNet50    → 82.94%
```


---

# 🍕 Supported Food Classes

The system supports **34 Food Categories**.

Some examples include:

* 🍕 Pizza
* 🍔 Burger
* 🫓 Butter Naan
* 🍩 Donut
* 🥪 Sandwich
* 🍣 Sushi
* 🌮 Taco
* 🥟 Samosa
* 🍛 Pav Bhaji
* 🥟 Momos
* 🥙 Paani Puri
* 🍯 Jalebi
* 🍨 Kulfi

And many more food categories.

---

# ✨ Application Features

The Streamlit application includes:

* 📤 Upload Food Image
* 🧠 Select Deep Learning Model
* 🤖 AI-Powered Food Classification
* 🎯 Prediction Confidence
* 🏆 Top 3 Predictions
* 📊 Model Accuracy Comparison
* 📈 Accuracy Visualization
* 🥇 Best Model Information
* 🕒 Prediction History
* 👩‍💻 Professional Developer Profile Section
* 🎨 Modern and Interactive User Interface

---

# 🖥️ Application Interface

The application is divided into multiple interactive sections.

## 👤 User Information

Displays developer information and primary technical skill.

## 🍔 Food Prediction

Users can:

1. Upload a food image.
2. Select a Deep Learning model.
3. Choose between CNN, VGG16, or ResNet50.
4. Run the prediction.
5. View the predicted food category.
6. Check prediction confidence.

## 🎯 Prediction Result

Displays:

* Predicted Food Category
* Prediction Confidence
* Confidence Level
* Top Predictions

## 🏆 Top 3 Predictions

The application displays the three most probable food categories predicted by the selected Deep Learning model.

## 📊 Model Comparison

Users can compare the validation accuracy of:

* Custom CNN
* VGG16
* ResNet50

---

# 🛠️ Technologies Used

| Technology        | Purpose                    |
| ----------------- | -------------------------- |
| Python            | Programming Language       |
| TensorFlow        | Deep Learning Framework    |
| Keras             | Neural Network Development |
| Streamlit         | Web Application            |
| NumPy             | Numerical Computing        |
| Pandas            | Data Processing            |
| Pillow            | Image Processing           |
| Deep Learning     | Image Classification       |
| Computer Vision   | Food Image Analysis        |
| Transfer Learning | Pre-trained Model Training |

---

# 📂 Project Structure

```text
Food_Selection_Using_Deep_Learning/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .gitattributes
│
├── models/
│   ├── food_classification_cnn.keras
│   ├── food_classification_vgg16.keras
│   └── food_classification_resnet50.keras
│
├── assets/
│   ├── profile.png
│   └── food_app_preview.png
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
├── test_images/
│
└── scripts/
```

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/SyedaNazneen/Food_Selection_Using_Deep_Learning.git
```

---

## 2️⃣ Navigate to Project Directory

```bash
cd Food_Selection_Using_Deep_Learning
```

---

## 3️⃣ Create a Virtual Environment

```bash
python -m venv .venv
```

---

## 4️⃣ Activate Virtual Environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

---

## 5️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will start locally in your browser.

---

# 📈 Model Performance

The trained Deep Learning models were evaluated using validation data.

## Final Results

| Model       | Accuracy   |
| ----------- | ---------- |
| Custom CNN  | 64.35%     |
| VGG16       | 75.12%     |
| 🏆 ResNet50 | **82.94%** |

🏆 **ResNet50 achieved the highest validation accuracy and was selected as the best-performing model.**

---

# 🧠 Key Learning Outcomes

Through this project, the following concepts were implemented and explored:

* Deep Learning
* Convolutional Neural Networks
* Transfer Learning
* Computer Vision
* Image Classification
* Model Comparison
* Model Evaluation
* Prediction Confidence
* Top-K Predictions
* Streamlit Deployment
* GitHub Project Deployment
* TensorFlow and Keras

---

# 🔮 Future Improvements

Possible future enhancements include:

* 📈 Increase Dataset Size
* 🍔 Add More Food Categories
* 🧠 Improve Model Accuracy
* 🤖 Implement Ensemble Learning
* 🔍 Add Grad-CAM Visualization
* 🌐 Add REST API Support
* 📱 Create a Mobile Application
* 📊 Add Advanced Model Analytics
* ⚡ Improve Prediction Speed
* ☁️ Deploy on Additional Cloud Platforms

---

# 👩‍💻 Developer

## **Syeda Nazneen**

**Deep Learning Developer | Python | Machine Learning | Computer Vision**

Passionate about building AI-powered applications and teaching Python, Machine Learning, Deep Learning, and Artificial Intelligence.

### GitHub

[https://github.com/SyedaNazneen](https://github.com/SyedaNazneen)

---

# ⭐ Support

If you like this project, please consider giving the repository a **⭐ Star on GitHub**.

Your support motivates me to build and share more AI and Deep Learning projects.

---

# 📄 License

This project is created for **Educational and Portfolio Purposes**.

