---
title: eSSVI
tags: ['technology', 'Finance', 'Quantitative Analysis']
categories: ['Quant']
description: 'A hands-on exploration of how eSSVI parameterizes volatility, what happens inside its optimization loop, and what each failure mode actually reveals about your data'
cover: ''
banner: ''
poster: ''
topic: ''
headline: 'Inside eSSVI: Beyond the Equations and into the Optimization Problem'
caption: 'What eSSVI really optimizes — and how to debug when it fails.'
color: ''
sticky: 0
mermaid: false
katex: false
mathjax: True
column: ''
author: ''
references: ''
comments: true
indexing: true
breadcrumb: true
h1: 'eSSVI: A Forensic Approach To Modelling Implied Volatilty'
type: tech
date: 2025-10-03 16:29:09
---
# Introduction

## 1. Stochastic Processes and the Heston Model

A **stochastic process** is a collection of random variables representing the evolution of a system over time under uncertainty.  
In finance, such processes are used to model the **random behavior of asset prices**, interest rates, and volatility.

### Example: The Heston Stochastic Volatility Model

The Heston model assumes that both the asset price and its variance evolve randomly over time, driven by correlated Brownian motions.
<div style="overflow-x: auto;">
$$
\begin{aligned}
dS_t &= \mu S_tdt + \sqrt{V_t}S_tdW_t^S ,\\ 
\end{aligned}
\begin{aligned}
dV_t &= \kappa(\theta - V_t)dt + \xi\sqrt{V_t}dW_t^V ,\\
\end{aligned}
$$
$$
\begin{aligned}
dW_t^SdW_t^V &= \rho dt
\end{aligned}
$$
</div>

### Explanation of Terms

| Symbol | Meaning |
|:--------|:--------|
| $$\(S_t\)$$ | Underlying asset price |
| $$\(V_t\)$$ | Instantaneous (stochastic) variance |
| $$\(\mu\)$$ | Drift of the asset price |
| $$\(\kappa\)$$ | Speed of mean reversion of variance |
| $$\(\theta\)$$ | Long-term mean of variance |
| $$\(\xi\)$$ | Volatility of volatility |
| $$\(\rho\)$$ | Correlation between the asset and variance processes |
| $$\(dW_t^S, dW_t^V\)$$ | Brownian motions driving randomness |

### Key Idea

The asset price $\(S_t\)$ evolves under **random shocks**, influenced by both its own dynamics and a **stochastic variance** $\(V_t\)$.  
This dual randomness captures features observed in real markets — most notably **volatility smiles and skews** in implied volatility surfaces derived from option prices.

## 2. How the Brownian Component Realizes During Training and Prediction (Heston Perspective)

The Brownian component in stochastic models like **Heston** represents the random source of uncertainty that drives both asset prices and volatility.  
When we “train” or calibrate the model, we aren’t fitting a single random path — we’re fitting the *distribution* implied by many such random paths.

### Step 1: Simulating Brownian Paths

Time is discretized into small intervals (e.g., 252 trading days in a year).  
Two correlated standard normal random variables $\( Z_S, Z_V \sim \mathcal{N}(0,1) \)$ are generated with correlation $\( \rho \)$.

Using **Euler discretization**:
<div style="overflow-x: auto;">

$$
\begin{aligned}
S_{t+\Delta t} &= S_t + rS_t\Delta t + \sqrt{V_t}S_t\sqrt{\Delta t}Z_S \\ \\ \\ \\
V_{t+\Delta t} &= V_t + \kappa(\theta - V_t)\Delta t + \xi\sqrt{V_t}\sqrt{\Delta t}Z_V \\
\text{with } \text{corr}(Z_S, Z_V) = \rho
\end{aligned}
$$
</div>
Running this for many simulated paths (e.g., 10,000) gives a *distribution* of terminal prices $\( S_T \)$.  
Each path is random, but the **statistical structure** across all paths reflects the model’s behavior.

### Step 2–4: Calibration and Comparison to Market

For each path, compute the option payoff:

$
C_i = e^{-rT} \max(S_T^{(i)} - K, 0)
$

The model price is the expected value:

$
C_{\text{model}} = \frac{1}{M}\sum_{i=1}^{M} C_i
$

Calibration adjusts parameters $\((\kappa, \theta, \xi, \rho, V_0)\)$ to minimize the total squared error versus market option prices:

$
\min_{\text{params}} \sum_{i,j} \left[C_{\text{model}}(K_i, T_j) - C_{\text{mkt}}(K_i, T_j)\right]^2
$

Typical optimizers include **BFGS**, **L-BFGS-B**, or global methods like **Differential Evolution**.

### Step 5: Forecasting with the Calibrated Model

Once calibrated, the model can be used to simulate new price distributions and compute expectations:

- $\( E[S_T] \)$: Expected price  
- Implied volatility distribution  
- Confidence intervals (e.g., 5%–95% quantiles of $\( S_T \)$)  
- Risk metrics like VaR and Greeks  

This process doesn’t predict exact future prices — it predicts the **distributional pattern** of randomness consistent with market-implied volatility.



