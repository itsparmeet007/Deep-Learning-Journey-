# Vanishing Gradient Problem (VGP)

## Overview

The **Vanishing Gradient Problem** is a common challenge in training deep neural networks. It occurs when gradients become extremely small as they propagate backwards through the network during backpropagation.

As a result, the weights in earlier layers receive very small updates, making it difficult for those layers to learn effectively.

This project demonstrates the Vanishing Gradient Problem using a deep neural network built with TensorFlow/Keras. It compares two activation functions:

- Sigmoid
- ReLU (Rectified Linear Unit)

The objective is to understand how activation functions influence gradient flow and neural network training.

## 1. Dataset

This experiment uses the `make_moons` dataset from Scikit-learn.

```python
from sklearn.datasets import make_moons

X, y = make_moons(
    n_samples=250,
    noise=0.2,
    random_state=42
)
```

**Dataset characteristics:**
- 250 samples.
- 2 input features.
- Binary classification.
- Non-linearly separable classes.

## 2. Neural Network Architecture

A deep neural network is created using Keras Sequential API.

Architecture:

- Input layer: 2 features.
- Hidden layers: 9 Dense layers.
- Neurons per hidden layer: 10.
- Output layer: 1 neuron.
- Loss function: Binary Crossentropy.
- Optimizer: Adam.
- Epochs: 100.
- Batch size: 10.

Two experiments are conducted by changing the hidden-layer activation function.

### Experiment 1: Sigmoid Activation

Sigmoid is used in all hidden layers and the output layer.

Mathematical representation:

\[
\sigma(x)=\frac{1}{1+e^{-x}}
\]

Derivative:

\[
\sigma'(x)=\sigma(x)(1-\sigma(x))
\]

The maximum derivative of sigmoid is 0.25.

When sigmoid receives very large positive or negative inputs, its derivative approaches zero. During backpropagation, repeatedly multiplying small derivatives can cause gradients to vanish.

### Experiment 2: ReLU Activation

ReLU replaces sigmoid in the hidden layers. The output layer continues using sigmoid for binary classification.

Mathematical representation:

\[
ReLU(x)=\max(0,x)
\]

Derivative:

\[
ReLU'(x)=
\begin{cases}
1 & x>0\\
0 & x<0
\end{cases}
\]

ReLU helps reduce the saturation-related vanishing gradient problem because its derivative is 1 for positive inputs.

However, inactive ReLU neurons can also experience zero gradients.

## 3. Understanding Gradient Norms

A gradient represents how the loss changes with respect to a model parameter.

\[
\frac{\partial L}{\partial W}
\]

Each Dense layer contains multiple weights, so we calculate the L2 norm to represent the overall magnitude of its weight gradients.

\[
||G||_2=\sqrt{\sum_i G_i^2}
\]

In TensorFlow:

```python
gradient_norm = tf.norm(kernel_gradient)
```

**Interpretation:**

- Large gradient norm: relatively large overall gradient magnitude.
- Small gradient norm: relatively small overall gradient magnitude.
- Extremely small gradients in earlier layers can indicate vanishing gradients.

Gradient norms should be compared using the same batch and model state.

## 4. Experimental Results

### Sigmoid Experiment

Observed gradient norms:

| Layer | Gradient norm |
|---|---:|
| dense_50 | 0.17571 |
| dense_51 | 0.18390 |
| dense_52 | 0.09947 |
| dense_53 | 0.05241 |
| dense_54 | 0.03035 |
| dense_55 | 0.01519 |
| dense_56 | 0.00842 |
| dense_57 | 0.00494 |
| dense_58 | 0.00599 |
| dense_59 | 0.02504 |

**Observation:**

Gradient magnitudes decrease substantially across several layers. This is consistent with gradient attenuation and motivates further investigation of vanishing gradients.

The loss initially decreases, experiences a plateau around 0.693, and later starts decreasing again.

### ReLU Experiment

Observed gradient norms:

| Layer | Gradient norm |
|---|---:|
| dense | 0.70992 |
| dense_1 | 1.07950 |
| dense_2 | 0.90896 |
| dense_3 | 0.97520 |
| dense_4 | 0.66897 |
| dense_5 | 0.54155 |
| dense_6 | 0.32779 |
| dense_7 | 0.18368 |
| dense_8 | 0.10295 |
| dense_9 | 0.06984 |
| dense_10 | 0.08073 |

**Observation:**

The ReLU experiment produces substantially different gradient magnitudes.

The training loss decreases initially, but also shows fluctuations and sudden spikes during training.

ReLU reduces saturation-related gradient shrinkage, but does not guarantee perfectly stable training or eliminate all vanishing gradients.

## 5. Visualizations

The following visualizations are used to understand model behaviour:

1. Training Loss vs Epochs — shows how the loss changes during training.
2. Gradient Norm vs Layers — shows how gradient magnitudes vary across the network.
3. Gradient Heatmap — provides a visual comparison of gradient magnitudes across layers.

These visualizations help investigate gradient flow and optimization behaviour.

## 6. Why Does the Vanishing Gradient Problem Matter?

When gradients become too small:

- Earlier layers learn very slowly.
- Weight updates become insignificant.
- Training deep neural networks becomes difficult.
- The network may struggle to learn useful representations.

## 7. Techniques to Reduce Vanishing Gradients

Some commonly used approaches include:

- ReLU and its variants, such as Leaky ReLU.
- Proper weight initialization (He or Xavier initialization).
- Batch Normalization.
- Residual connections (ResNet).
- LSTM and GRU architectures for sequence modelling.

## 8. Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## 9. Conclusion

This project explores how activation functions influence gradient propagation in deep neural networks.

The sigmoid experiment demonstrates substantial reductions in gradient magnitude across several layers, while the ReLU experiment shows different gradient behaviour.

The experiments highlight why activation-function selection matters when training deep neural networks.

Gradient norms, training-loss curves and visualizations provide complementary ways to investigate the Vanishing Gradient Problem.

## References

- [Google Machine Learning Crash Course — Backpropagation](https://developers.google.com/machine-learning/crash-course/neural-networks/backpropagation)
- [Stanford CS224d — Vanishing Gradients Notebook](https://web.stanford.edu/class/cs224d/notebooks/vanishing_grad_example.html)
- [Keras — Activation Functions](https://keras.io/api/layers/activations/)

---

**Author:** Parmeet Singh

**Project:** Vanishing Gradient Problem (VGP)
