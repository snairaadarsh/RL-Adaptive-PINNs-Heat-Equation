# RL-Adaptive PINNs for Solving the 1D Heat Equation

This repository presents a Reinforcement Learning (RL) based adaptive sampling framework for Physics-Informed Neural Networks (PINNs) applied to solving the one-dimensional heat equation. The proposed approach addresses the inefficiency of traditional PINNs that rely on uniform sampling of collocation points during training.

An Upper Confidence Bound (UCB) multi-armed bandit algorithm is integrated into the PINN training process to adaptively guide collocation point selection by focusing on regions with higher residual errors.

Code will be released after manuscript publication.
Manuscript accepted at the **2026 4th International Conference on Integrated Circuits and Communication Systems (ICICACS)**.


Problem Overview
----------------
Physics-Informed Neural Networks solve partial differential equations by embedding physical laws into the neural network loss function. However, traditional PINNs use uniform sampling of collocation points, which leads to inefficient learning in regions that are harder to approximate and slower convergence.

This work proposes an RL-based adaptive sampling strategy to overcome this limitation by dynamically allocating more collocation points to regions with higher residual errors.


Governing Equation
------------------
The one-dimensional heat equation is used as the benchmark problem:

```
∂u(x,t)/∂t = α ∂²u(x,t)/∂x²
```

with `x ∈ [0, 1]`, `t ≥ 0`, initial condition `u(x, 0) = sin(πx)`, and boundary conditions `u(0, t) = u(1, t) = 0`.

The objective is to approximate the solution using a neural network trained with a physics-informed loss that enforces the governing PDE along with initial and boundary conditions.


Methodology
-----------
Two training strategies are compared:

1. PINN with uniform random sampling of collocation points.
2. PINN with RL-based adaptive sampling using the UCB algorithm.

Both approaches use the same network architecture, number of epochs, and hyperparameter settings to ensure a fair comparison.


PINN Architecture
-----------------
The best-performing network configuration identified through architecture search:

- **Layers (L):** 3
- **Width (W):** 50 neurons per layer
- **Activation function:** GELU
- **Inputs:** spatial and temporal coordinates (x, t)
- **Output:** temperature field u(x, t)

Loss function components:
- **PDE Residual Loss:** enforces the heat equation via automatic differentiation
- **Initial Condition Loss:** enforces `u(x, 0) = sin(πx)`
- **Boundary Condition Loss:** enforces `u(0, t) = u(1, t) = 0`

Architecture search results (final L2 error):

| Architecture   | L | W   | Activation | Final L2 Error |
|----------------|---|-----|------------|----------------|
| L3_W50_gelu    | 3 | 50  | GELU       | **0.019244**   |
| L4_W20_tanh    | 4 | 20  | tanh       | 0.025709       |
| L4_W50_tanh    | 4 | 50  | tanh       | 0.027705       |
| L5_W50_tanh    | 5 | 50  | tanh       | 0.044664       |
| L3_W100_tanh   | 3 | 100 | tanh       | 0.045242       |
| L3_W20_tanh    | 3 | 20  | tanh       | 0.068563       |
| L1_W50_tanh    | 1 | 50  | tanh       | 0.674835       |
| L3_W50_relu    | 3 | 50  | ReLU       | 3.554516       |


Reinforcement Learning Based Adaptive Sampling
----------------------------------------------
Reinforcement learning is used to guide the selection of collocation points during training.

- **State (S):** vector of segment-wise mean PDE residuals across the domain
- **Action (A):** selection of spatial segments for sampling new collocation points
- **Reward (R):** log-transformed residual reduction after training on selected points — `r_i = log(1 + R̄_i)`

The domain `[0, 1] × [0, T]` is partitioned into **N = 8** equal segments, each acting as an independent arm in the multi-armed bandit framework. The UCB score for each segment is computed as:

```
UCB_i = r̄_i + c * sqrt(ln(N_total) / n_i)
```

where `r̄_i` is the average reward, `n_i` is the number of times segment `i` was selected, `N_total` is the total selections so far, and `c` is the exploration constant.


UCB Parameter Sensitivity
--------------------------
An ablation study was performed over four values of the exploration constant `c`:

| UCB Constant (c) | Final L2 Error | Training Time (s) |
|------------------|----------------|-------------------|
| 0.5              | 0.012640       | 26.16             |
| 1.0              | 0.012057       | 26.87             |
| **2.0**          | **0.007348**   | 28.18             |
| 4.0              | 0.009624       | 28.03             |

**c = 2** was selected for all reported experiments, yielding the lowest final L2 error of 0.007348.


Training Setup
--------------
Both models are trained under identical conditions:

- **Epochs:** 3000
- **Optimizer:** Adam
- **Same network architecture** for fair comparison

The UCB algorithm is invoked at each iteration to select new collocation points based on residual feedback.


Evaluation Metrics
------------------
Performance is compared using:

- **L2 Error** — accuracy against the analytical solution
- **Final Loss** — how well physics constraints are satisfied
- **Average PDE Residual** — error distribution across the domain
- **Training Time** — computational cost
- **Convergence Behavior** — how quickly the model reaches threshold accuracy

Residual heatmaps, sampling distribution, and UCB arm selection trends are used to analyze training behavior.


Results Summary
---------------
Comparative analysis across four independent runs:

| Run | Method       | Final L2 Error | Training Time (s) | Final Loss | Avg Residual | L2 Improvement |
|-----|--------------|----------------|-------------------|------------|--------------|----------------|
| 1   | Uniform      | 0.029202       | 24.45             | 0.000358   | 0.006614     | 60.15%         |
|     | UCB-Adaptive | 0.011638       | 24.38             | 0.000059   | 0.004046     |                |
| 2   | Uniform      | 0.024237       | 27.01             | 0.000141   | 0.005840     | 57.02%         |
|     | UCB-Adaptive | 0.010418       | 26.60             | 0.000028   | 0.002090     |                |
| 3   | Uniform      | 0.020102       | 26.88             | 0.000134   | 0.005724     | 53.18%         |
|     | UCB-Adaptive | 0.009411       | 26.35             | 0.000046   | 0.002988     |                |
| 4   | Uniform      | 0.023157       | 24.56             | 0.000152   | 0.004757     | 62.80%         |
|     | UCB-Adaptive | 0.008615       | 26.75             | 0.000052   | 0.003758     |                |

The UCB-based adaptive PINN achieves **53.18% to 62.80% reduction in L2 error** across all runs compared to uniform sampling, while maintaining comparable training times.


Conclusion
----------
This work demonstrates that integrating reinforcement learning with Physics-Informed Neural Networks through a UCB-based adaptive sampling strategy significantly improves solution accuracy without increasing computational cost.

By treating collocation point selection as a multi-armed bandit problem, the framework dynamically prioritizes high-residual regions based on PDE residual feedback. The ablation study confirmed c = 2 as the optimal exploration constant, and architecture experiments validated the chosen network configuration (L=3, W=50, GELU).


Future Work
-----------
- Extension to higher-dimensional PDEs
- Application to coupled or nonlinear PDE systems
- Integration with more advanced RL algorithms
- Adaptive sampling for multi-physics PINNs


Authors
-------
- Aadarsh S Nair — Department of Mathematics, Amrita Vishwa Vidyapeetham, Coimbatore
- Subburaj — Department of Mathematics, Amrita Vishwa Vidyapeetham, Coimbatore
- Saran P P — Department of Mathematics, Amrita Vishwa Vidyapeetham, Coimbatore
- Rishab P Nambiar — Department of Mathematics, Amrita Vishwa Vidyapeetham, Coimbatore
- Abishek S — Department of Mathematics, Amrita Vishwa Vidyapeetham, Coimbatore


Status
------
Paper accepted at the **2026 4th International Conference on Integrated Circuits and Communication Systems (ICICACS)**.
