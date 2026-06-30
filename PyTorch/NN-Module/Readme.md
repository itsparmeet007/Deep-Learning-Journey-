# 🧠 PyTorch Training Pipeline using `nn.Module`

> A beginner-friendly implementation of a Neural Network using PyTorch's `nn.Module` API on the Breast Cancer Wisconsin Dataset.

---

## 📌 Overview

This notebook demonstrates how to build and train a simple Feedforward Neural Network (FNN) using PyTorch's `nn.Module`.

Unlike the previous notebook where the network was built manually (using custom weights, bias, and gradient updates), this notebook introduces PyTorch's high-level neural network API to simplify model development while leveraging Automatic Differentiation (Autograd).

The notebook focuses on understanding the complete training workflow rather than building a complex model.

---

## 🎯 Learning Objectives

After completing this notebook, you will understand:

- Creating custom models using `nn.Module`
- Understanding how `nn.Linear` works
- Forward propagation using the `forward()` method
- Hidden layers and output layers
- ReLU activation function
- Sigmoid activation function
- Building models using `nn.Sequential`
- Binary Cross Entropy Loss (`BCELoss`)
- Optimizers (`torch.optim`)
- Complete Training Pipeline
- Model Evaluation
- Difference between manual implementation and `nn.Module`

---

## 🧠 Neural Network Architecture

```text
Input Features
      │
      ▼
Linear Layer (Input → Hidden)
      │
      ▼
ReLU Activation
      │
      ▼
Linear Layer (Hidden → Output)
      │
      ▼
Sigmoid Activation
      │
      ▼
Prediction (0–1)
```

---

## 📂 Dataset

**Breast Cancer Wisconsin Dataset**

- Binary Classification Problem
- Predict whether a tumor is:
  - Benign (0)
  - Malignant (1)

---

## 🛠️ Technologies Used

- Python
- PyTorch
- NumPy
- Scikit-learn
- Google Colab / Kaggle Notebook

---

## 📚 Concepts Covered

### Neural Networks
- Input Layer
- Hidden Layer
- Output Layer

### PyTorch Modules
- `nn.Module`
- `nn.Linear`
- `nn.Sequential`
- `forward()`

### Activation Functions
- ReLU
- Sigmoid

### Training Pipeline
- Forward Pass
- Loss Calculation
- Backpropagation
- Optimizer Step
- Gradient Reset (`zero_grad()`)

### Loss Function
- Binary Cross Entropy (BCELoss)

### Optimizer
- Stochastic Gradient Descent (SGD)

---

## 🚀 Workflow

```text
Load Dataset
      │
      ▼
Convert to PyTorch Tensors
      │
      ▼
Build Neural Network
      │
      ▼
Define Loss Function
      │
      ▼
Define Optimizer
      │
      ▼
Training Loop
      │
      ├── Forward Pass
      ├── Compute Loss
      ├── Backward Pass
      ├── Optimizer Step
      └── Zero Gradients
      │
      ▼
Model Evaluation
```

---

## 📖 Key Takeaways

- Learned how PyTorch manages model parameters automatically.
- Understood the purpose of `forward()` in neural networks.
- Explored the difference between manual neural network implementation and `nn.Module`.
- Built multi-layer neural networks using `nn.Sequential`.
- Learned how optimizers update parameters automatically.
- Implemented a complete binary classification training pipeline.

---

## 🔜 Next Steps

- Deep Neural Networks
- Different Optimizers (Adam, RMSProp)
- Loss Functions
- Convolutional Neural Networks (CNNs)
- Transfer Learning

---

## 👨‍💻 About Me

**Parmeet Singh**

Computer Science Student | Machine Learning & Deep Learning Enthusiast

Currently documenting my complete Deep Learning learning journey through practical implementations using PyTorch.

- 🌟 GitHub: *Add your GitHub Profile Link*
- 💼 LinkedIn: *Add your LinkedIn Profile Link*

---

⭐ If you found this notebook helpful, consider giving this repository a **Star**!
