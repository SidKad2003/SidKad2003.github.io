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
In this post, I take a forensic approach to implementing eSSVI — documenting the errors I encountered and how I diagnosed and resolved them.
# Stochastic and Stochastic Volatility Inspired
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
| $$S_t$$ | Underlying asset price |
| $$V_t$$ | Instantaneous (stochastic) variance |
| $$\mu$$ | Drift of the asset price |
| $$\kappa$$ | Speed of mean reversion of variance |
| $$\theta$$ | Long-term mean of variance |
| $$\xi$$ | Volatility of volatility |
| $$\rho$$ | Correlation between the asset and variance processes |
| $$dW_t^S, dW_t^V$$ | Brownian motions driving randomness |

### Key Idea

The asset price $S_t$ evolves under **random shocks**, influenced by both its own dynamics and a **stochastic variance** $V_t$.  
This dual randomness captures features observed in real markets — most notably **volatility smiles and skews** in implied volatility surfaces derived from option prices.

## 2. How the Brownian Component Realizes During Training and Prediction (Heston Perspective)

The Brownian component in stochastic models like **Heston** represents the random source of uncertainty that drives both asset prices and volatility.  
When we “train” or calibrate the model, we aren’t fitting a single random path — we’re fitting the *distribution* implied by many such random paths.

### Step 1: Simulating Brownian Paths

Time is discretized into small intervals (e.g., 252 trading days in a year).  
Two correlated standard normal random variables $\( Z_S, Z_V \sim \mathcal{N}(0,1) \)$ are generated with correlation $\rho$.

Using **Euler discretization**:
<div style="overflow-x: auto;">
$$
\begin{aligned}
S_{t+\Delta t} &= S_t + rS_t\Delta t + \sqrt{V_t}S_t\sqrt{\Delta t}Z_S \\ 
\end{aligned}
$$
$$
\begin{aligned}
V_{t+\Delta t} &= V_t + \kappa(\theta - V_t)\Delta t + \xi\sqrt{V_t}\sqrt{\Delta t}Z_V \\
\end{aligned}
\begin{aligned}
\text{with } \text{corr}(Z_S, Z_V) = \rho
\end{aligned}
$$
</div>

Running this for many simulated paths (e.g., 10,000) gives a **Distribution** of terminal prices $S_t$.  
Each path is random, but the **Statistical Structure** across all paths reflects the model’s behavior.
### Step 2–4: Calibration and Comparison to Market

For each path, compute the option payoff: $  C_i = e^{-rT} \max(S_T^{(i)} - K, 0)$

The model price is the expected value: $  C_{\text{model}} = \frac{1}{M}\sum_{i=1}^{M} C_i$

Calibration adjusts parameters $(\kappa, \theta, \xi, \rho, V_0)$ to minimize the total squared error versus market option prices:
$$\min_{\text{params}} \sum_{i,j} \left[C_{\text{model}}(K_i, T_j) - C_{\text{mkt}}(K_i, T_j)\right]^2$$

Typical optimizers include **BFGS**, **L-BFGS-B**, or global methods like **Differential Evolution**.

### Step 5: Forecasting with the Calibrated Model

Once calibrated, the model can be used to simulate new price distributions and compute expectations:

- $E[S_T]$: Expected price  
- Implied volatility distribution  
- Confidence intervals (e.g., 5%–95% quantiles of $S_T$)  
- Risk metrics like VaR and Greeks  

Hence, this process doesn’t predict exact future prices — it predicts the **distributional pattern** of randomness consistent with market-implied volatility.

As a result the Heston model helped researchers understand phenomena such as skew, smile, and the term structure of implied volatility by providing a stochastic volatility framework that explains these features more realistically than simpler models like Black-Scholes. It was among the first models able to capture these effects with a semi-analytical solution, making it useful for practical implementation.

Based on the understanding from such stochastic volatility models, researchers developed parametric models like SVI and later eSSVI to directly parametrize and fit the observed implied volatility surface. These models make calibration more efficient and ensure no-arbitrage constraints while preserving the realistic shapes (smile, skew, term structure) originally explored by models such as Heston. Thus, SSVI and eSSVI was built as a natural evolution to efficiently capture the structure that stochastic volatility models helped uncover.
# 3. eSSVI
The elegance of the eSSVI model lies in its ability to capture the full implied volatility surface with just a few parameters, while staying arbitrage-free. But implementing it in practice is far from plug-and-play.

In this section, I break down the inner mechanics of eSSVI — from what each parameter does, to how the surface is calibrated as an optimization problem. I also dive into the errors I faced during implementation, what caused them, and how I fixed them.

This is where theory meets reality.
## Understanding the eSSVI Parametrization

The eSSVI model defines the **implied total variance** surface \( w(k, t) \), which is the squared implied volatility multiplied by maturity. Its formulation is:

<div style="overflow-x: auto;">
$$
\begin{aligned}
w(k, t) = \frac{\theta_t}{2} \left\{ 1 + \rho_t \varphi_t k + \sqrt{ (\varphi_t k + \rho_t)^2 + (1 - \rho_t^2) } \right\} 
\end{aligned}
$$
</div>

### What Each Symbol Means

| Symbol | Meaning |
|--------|---------|
| $$w(k, t)$$ | Total implied variance at log-moneyness $k$ and maturity $t$. This is  $\sigma_{\text{BS}}^2(k, t) \cdot t$, where $\sigma_{\text{BS}}$ is Black-Scholes implied volatility. |
| $$k$$ | **Log-moneyness**, defined as $\log(K/F_t)$, where $K$ is strike and $F_t$ is forward price at maturity  $t$. |
| $$t$$ | **Time to maturity** (in years). |
| $$\theta_t$$ | **ATM total variance** at maturity $t$, i.e., $\sigma_{\text{ATM}}^2(t) \cdot t $. This anchors the surface at-the-money. |
| $$ \varphi_t $$ | **Slope control parameter** at maturity $t$. It governs the steepness of the volatility smile (w.r.t. $k$). |
| $$\rho_t$$ | **Skewness parameter** at maturity $t$. It controls asymmetry (skew) of the smile. Ranges between $-1$ and $1$. |

<!-- #### 🧩 Interpretation:

- The term $\rho_t \varphi_t k$ adds **linear skew**.
- The square root term introduces **curvature**, ensuring a smooth and arbitrage-free shape.
- The overall structure ensures that **smiles and skews** observed in market implied volatilities can be fit accurately and consistently over strikes and maturities.
- Since, eSSVI (along SVI and SSVI) is a parameteric model, each of  $\rho_t$ and $\varphi_t$ capture assigned characteristics. -->

### 🧾 Role of Each Symbol in the eSSVI Model

#### θₜ — ATM Total Variance

The symbol $\theta_t$ represents the total implied variance at-the-money (ATM) for a given maturity $t$. This is usually computed as:$\theta_t = \sigma_{\text{ATM}}^2 \cdot t$

- This parameter serves as the **vertical anchor** for the volatility surface. It reflects the market's consensus on average volatility over the period $[0, t]$, and is typically the most stable and liquid point in the volatility surface. Since it comes directly from market data, no fitting is required for $\theta_t$.
- The ATM total implied variance θ is considered one of the most stable, liquid, and reliable quotes in the options market. Both SSVI and eSSVI "anchor" their volatility smile parameterizations to exactly match the ATM variance, because this reduces model fit uncertainty and noise. Each maturity slice is characterized first by its ATM variance, then by additional parameters to capture skew/curvature. This anchoring ensures the smile passes through the most trusted market point, minimizing extrapolation error at-the-money.
- The $\frac{\theta_t}{2} \left\{ 1 + \rho_t \varphi_t k + \sqrt{ (\varphi_t k + \rho_t)^2 + (1 - \rho_t^2) } \right\}$ term transform the ATM variance as a funtion of **Log Moneyness** and **Time till Expiry**.

---

#### k — Log-Moneyness

Log-moneyness $k$ is defined as:$k = \log\left(\frac{K}{F_t}\right)$ where $K$ is the strike price and $F_t$ is the forward price at maturity$t$, computed as:$F_t = S_0 e^{rt}$

- Using $k$ instead of raw strike removes scale effects and normalizes the surface across maturities. It allows the model to operate on a consistent, dimensionless domain regardless of the underlying asset's level.

---

#### ρₜ — Skew Parameter

The skew parameter $\rho_t$ controls the **asymmetry** of the implied volatility smile. When:

- $\rho_t < 0$: the smile has **left skew** (higher vol for OTM puts)
- $\rho_t > 0$: the smile has **right skew** (higher vol for OTM calls)

This parameter is commonly modeled as a **linear function of time**:$\rho_t = A_\rho t + B_\rho$ .This allows the skew to evolve smoothly with maturity. It must satisfy $|\rho_t| < 1$ to ensure the model remains arbitrage-free.

---

#### φₜ — Curvature / Scale Parameter

The curvature parameter $\varphi_t$ determines the **steepness** and **curvature** of the volatility smile beyond the ATM point. 
- It is not directly calibrated, but is computed from:$\varphi_t = \eta \cdot (\rho_t)^{-\gamma}$. 
- Here, \( \eta \) and \( \gamma \) are themselves modeled as **linear functions of time**: $\eta = A_\eta t + B_\eta, \quad \gamma = A_\gamma t + B_\gamma$

This form provides the flexibility to shape the smile appropriately across maturities. The combination of $\rho_t$ and $\varphi_t$ enables the model to reproduce both the skew and the smile seen in market data.

---

### Intuition: Transforming ATM Variance Across Strikes and Time

The full eSSVI formula adjusts the ATM total variance $\theta_t$ into a surface that varies with strike (via $k$):

$$
\begin{align}
w(k, t) = \frac{\theta_t}{2} \left\{ 1 + \rho_t \varphi_t k + \sqrt{ (\varphi_t k + \rho_t)^2 + (1 - \rho_t^2) } \right\}
\end{align}
$$

- The **linear term** $\rho_t \varphi_t k$ introduces asymmetry.
- The **square root term** introduces smooth curvature, ensuring the surface bends around ATM.

Together, they produce a total variance surface that:
- Matches market smiles and skews.
- Respects no-arbitrage constraints.
- Evolves smoothly with both strike and maturity.

This structure is why eSSVI is widely used for **volatility surface calibration**: it's flexible, interpretable, and can be constrained to remain arbitrage-free with the right parameter settings.

This is what makes eSSVI powerful: it provides a flexible yet arbitrage-free **parametrization** of the implied volatility surface.

### 