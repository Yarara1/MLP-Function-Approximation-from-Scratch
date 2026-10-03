# MLP Function Approximation with Radial Fourier Features

This project implements a **single-hidden-layer MLP from scratch in MATLAB** to approximate a nonlinear 2D function.

The network uses `tanh` activation and is trained with **mini-batch gradient descent**. The project studies how hidden-layer size, batch size, learning rate, and Fourier feature encoding affect approximation performance.

---

## Model

<img width="361" height="321" alt="image" src="https://github.com/user-attachments/assets/ae991351-df67-4cb8-b9b1-d8a9a1c0b0a9" />


The model was implemented without MATLAB's built-in neural network training functions, including manual forward propagation and backpropagation.

---

## Hidden Layer Experiment

Different hidden-layer sizes were tested.

<img width="815" height="253" alt="image" src="https://github.com/user-attachments/assets/1f6681be-febb-4367-bf82-d71ae46a76c7" />


The best result was obtained with **150 hidden neurons**.

---

## Hyperparameter Tuning

Several batch-size and learning-rate combinations were also compared.

<img width="814" height="211" alt="image" src="https://github.com/user-attachments/assets/4becb882-64a5-4047-9a4c-6765578743a4" />


The best configuration used:

```text
Batch Size = 128
Learning Rate = 5e-4
```

---

## Radial Fourier Features

The standard MLP captured the overall shape of the target function but struggled with its high-frequency concentric patterns.

To improve this, radial Fourier features such as

```text
sin(ωr), cos(ωr), cos(θ), sin(θ)
```

were added to the input representation.
<img width="803" height="169" alt="image" src="https://github.com/user-attachments/assets/a95bb468-4c32-4570-af4d-3c9daa71f7b2" />


Using **K = 4** significantly improved the approximation.

---

## Key Result

```text
Standard MLP Validation RMSE : 0.2206
Fourier Feature MLP          : 0.0912
```

Radial Fourier features allowed the network to represent the oscillatory structure much more accurately.

---

## Gradient Check

Analytical gradients were compared with finite-difference approximations to verify the manually implemented backpropagation.

Gradient norms remained stable during training, with no major vanishing or exploding gradient problems.

---


## Technologies

`MATLAB` · `Neural Networks` · `Mini-Batch Gradient Descent` · `Backpropagation` · `Fourier Features` · `Regression`
