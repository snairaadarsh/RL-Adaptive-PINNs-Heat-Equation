# RL-Adaptive PINNs for Solving the 1D Heat Equation

This repository presents a Reinforcement Learning (RL) based adaptive sampling framework for Physics-Informed Neural Networks (PINNs) applied to solving the one-dimensional heat equation. The proposed approach addresses the inefficiency of traditional PINNs that rely on uniform sampling of collocation points during training.

An Upper Confidence Bound (UCB) multi-armed bandit algorithm is integrated into the PINN training process to adaptively guide collocation point selection by focusing on regions with higher residual errors.

Code will be released after manuscript publication.
Manuscript is currently under preparation.


Problem Overview
----------------
Physics-Informed Neural Networks solve partial differential equations by embedding physical laws into the neural network loss function. However, traditional PINNs use uniform sampling of collocation points, which leads to inefficient learning in regions that are harder to approximate and slower convergence.

This work proposes an RL-based adaptive sampling strategy to overcome this limitation by dynamically allocating more collocation points to regions with higher residual errors.


Governing Equation
------------------
The one-dimensional heat equation is used as the benchmark problem for this study.

The objective is to approximate the solution using a neural network trained with a physics-informed loss that enforces the governing PDE along with initial and boundary conditions.


Methodology
-----------
Two training strategies are compared:

1. PINN with uniform random sampling of collocation points.
2. PINN with RL-based adaptive sampling using the UCB algorithm.

Both approaches use the same network architecture, number of epochs, and hyperparameter settings to ensure a fair comparison.


PINN Architecture
-----------------
- Fully connected feedforward neural network
- Inputs: spatial and temporal coordinates (x, t)
- Output: temperature field u(x, t)
- Loss function components:
  - PDE residual loss
  - Initial condition loss
  - Boundary condition loss


Reinforcement Learning Based Adaptive Sampling
----------------------------------------------
Reinforcement learning is used to guide the selection of collocation points during training.

- State: current distribution of PDE residual errors
- Action: selection of spatial regions for sampling new collocation points
- Reward: reduction in residual error after training on selected points

The Upper Confidence Bound (UCB) algorithm balances exploration and exploitation by prioritizing regions with higher error while still ensuring domain coverage.


Training Setup
--------------
Both models are trained under identical conditions:

- Number of epochs: 3000
- Training time: approximately 70.5 seconds
- Same network architecture and optimizer settings

The UCB algorithm is invoked at each iteration to select new collocation points based on residual feedback.


Evaluation Metrics
------------------
The performance of uniform and RL-adaptive PINNs is evaluated using:

- L2 error
- Final loss value
- Average PDE residual
- Training time
- Convergence behavior


Results Summary
---------------
Uniform Sampling:
- Final L2 error: 0.033735
- Training time: 70.54 seconds
- Final loss: 0.000884
- Average residual: 0.014815

UCB-Adaptive Sampling:
- Final L2 error: 0.024194
- Training time: 70.48 seconds
- Final loss: 0.000434
- Average residual: 0.010954

The UCB-based adaptive PINN achieves a 28.28 percent reduction in L2 error compared to uniform sampling while maintaining similar training time and convergence epochs.


Conclusion
----------
This work demonstrates that integrating reinforcement learning with Physics-Informed Neural Networks through a UCB-based adaptive sampling strategy significantly improves solution accuracy without increasing computational cost.

By focusing training on regions with higher residual errors, the RL-enhanced PINN achieves better convergence and more efficient use of collocation points.


Future Work
-----------
- Extension to higher-dimensional PDEs
- Application to coupled or nonlinear PDE systems
- Integration with more advanced RL algorithms
- Adaptive sampling for multi-physics PINNs


Status
------
Paper under final preprartion

----
The complete manuscript, experimental code, trained models, and result visualizations will be made publicly available upon publication.
