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
# eSSVI
The elegance of the eSSVI model lies in its ability to capture the full implied volatility surface with just a few parameters, while staying arbitrage-free. But implementing it in practice is far from plug-and-play.

In this section, I break down the inner mechanics of **Power style** eSSVI — from what each parameter does, to how the surface is calibrated as an optimization problem. I also dive into the errors I faced during implementation, what caused them, and how I fixed them.

This is where theory meets reality.
## 1. Understanding the eSSVI Parametrization

The eSSVI model defines the **implied total variance** surface \( w(k, t) \), which is the squared implied volatility multiplied by maturity. Its formulation is:

<div style="overflow-x: auto;">
$$
\begin{aligned}
w(k, t) = \frac{\theta_t}{2} \left\{ 1 + \rho_t \varphi_t k + \sqrt{ (\varphi_t k + \rho_t)^2 + (1 - \rho_t^2) } \right\} 
\end{aligned}
$$
</div>


This is implemented in `total_variance()`:

```python
def total_variance(self, x, tind, kind):
    expiry = self._T[tind]
    k_val = self._k[expiry][kind]
    theta_val = self._theta[expiry]
    rho, p1, p2 = self._compute_parameters(x, expiry)

    if self._type == 'Heston':
        lam = p1
        x_val = lam * theta_val
        phi = (1 - np.exp(-x_val)) / x_val if x_val > 1e-5 else 1 - x_val / 2
    elif self._type == 'Power':
        eta, gamma = p1, p2
        phi = eta * (theta_val ** -gamma)

    term = phi * k_val + rho
    val = max(term**2 + (1 - rho**2), 1e-10)
    return theta_val / 2 * (1 + rho * phi * k_val + np.sqrt(val))
```
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

### Role of Each Symbol in the eSSVI Model

#### θₜ: ATM Total Variance

The symbol $\theta_t$ represents the total implied variance at-the-money (ATM) for a given maturity $t$. This is usually computed as: $\theta_t = \sigma_{\text{ATM}}^2 \cdot t$

- This parameter serves as the **vertical anchor** for the volatility surface. It reflects the market's consensus on average volatility over the period $[0, t]$, and is typically the most stable and liquid point in the volatility surface. Since it comes directly from market data, no fitting is required for $\theta_t$.
- The ATM total implied variance θ is considered one of the most stable, liquid, and reliable quotes in the options market. Both SSVI and eSSVI **Anchor** their volatility smile parameterizations to exactly match the ATM variance, because this reduces model fit uncertainty and noise. Each maturity slice is characterized first by its ATM variance, then by additional parameters to capture skew/curvature. This anchoring ensures the smile passes through the most trusted market point, minimizing extrapolation error at-the-money.
<!-- - The 

$$
\begin{aligned}
\frac{\theta_t}{2} \left\{ 1 + \rho_t \varphi_t k + \sqrt{ (\varphi_t k + \rho_t)^2 + (1 - \rho_t^2) } \right\}
\end{aligned}
$$
term transform the ATM variance as a funtion of **Log Moneyness** and **Time till Expiry**. -->

---

#### k: Log-Moneyness

Log-moneyness $k$ is defined as: $k = \log\left(\frac{K}{F_t}\right)$ where $k$ is the strike price and $F_t$ is the forward price at maturity $t$, computed as: $F_t = S_0 e^{-rt}$

- Using $k$ instead of raw strike removes scale effects and normalizes the surface across maturities. It allows the model to operate on a consistent, dimensionless domain regardless of the underlying asset's level.

---

#### ρₜ: Skew Parameter

The skew parameter $\rho_t$ controls the **asymmetry** of the implied volatility smile. When:

- $\rho_t < 0$: the smile has **left skew** (higher vol for OTM puts)
- $\rho_t > 0$: the smile has **right skew** (higher vol for OTM calls)

- Commonly modeled as a **linear function of time** : $\rho_t = A_\rho t + B_\rho$ .
- This allows the skew to evolve smoothly with maturity. It must satisfy $|\rho_t| < 1$ to ensure the model remains arbitrage-free.

---

#### φₜ: Curvature / Scale Parameter

The curvature parameter $\varphi_t$ determines the **steepness** and **curvature** of the volatility smile beyond the ATM point. 
- It is not directly calibrated, but is computed from: $\varphi_t = \eta \cdot (\rho_t)^{-\gamma}$. 
- Here, $\eta$ and $\gamma$ are themselves modeled as **linear functions of time**:

$$
\begin{aligned}
\eta = A_\eta t + B_\eta, \quad \gamma = A_\gamma t + B_\gamma
\end{aligned}
$$

This form provides the flexibility to shape the smile appropriately across maturities. The combination of $\rho_t$ and $\varphi_t$ enables the model to reproduce both the skew and the smile seen in market data.

---
Parameters were calculated in `_compute_parameters()`, where X is the array of parameters to be optimized.
```python
def _compute_parameters(self, x, T_val):
    raw_rho = x[0] + x[1] * T_val
    rho = raw_rho  # Can be clipped later if needed

    if self._type == 'Heston':
        lam = x[2] + x[3] * T_val
        return rho, lam, None
    elif self._type == 'Power':
        eta = x[2] + x[3] * T_val
        gamma = x[4] + x[5] * T_val
        return rho, eta, gamma
```
#### Arbitrage Constraints

These constraints ensure the **convexity of the total variance smile** for each maturity slice — i.e., **no butterfly arbitrage**. Derived from the work of Gatheral & Jacquier, they apply per maturity:

##### 1. $1 - |\rho| - 1 \times 10^{-5} > 0$
- **Type:** Butterfly
- **Meaning:** $\rho \in (-1, 1)$
- **Why it matters:** Keeps the square root in the SVI formula real and ensures skew doesn't lead to a non-convex smile.

##### 2. $\eta - 1 \times 10^{-5} > 0$
- **Type:** Butterfly
- **Meaning:** $\eta > 0$
- **Why it matters:** Ensures the curvature parameter is positive, giving the smile its shape. Negative curvature would flip the smile, creating arbitrage.

##### 3. $0.5 - |\gamma - 0.5| > 0$
- **Type:** Butterfly
- **Meaning:** $\gamma \in (0, 1)$ — commonly constrained tighter to $(0, 0.5]$
- **Why it matters:** Controls how the curvature scales with maturity. If $\gamma \notin (0,1)$, the surface becomes unstable or ill-formed.

##### 4. $2 - \eta (1 + |\rho|) > 0$
- **Type:** Butterfly
- **Meaning:** $\eta (1 + |\rho|) < 2$
- **Why it matters:** Known as the **Gatheral–Jacquier upper bound** — it prevents the total variance from becoming too steep in the wings, which would violate convexity.

---

#### Calendar Spread Arbitrage Constraints (Across Time)

These ensure the surface is consistent **across maturities**, i.e., there's **no calendar spread arbitrage**.

##### 5. $\theta_{t}$ must be strictly increasing in $t$
- **Type:** Calendar
- **Why it matters:** The ATM total variance should grow with maturity — declining variance implies the market expects less uncertainty further out, which contradicts financial intuition.

##### 6. $\psi(\theta_t)$ must be non-decreasing in $t$
- **Type:** Calendar
- **Why it matters:** Ensures the **overall steepness** of the volatility smile doesn't drop as maturity increases. Prevents arbitrage from crossing maturity slices.

##### 7. Slope continuity condition between maturities:
<div style="overflow-x: auto;">
$$
\begin{aligned}
\left| \frac{\rho_{i+1} \psi_{i+1} - \rho_i \psi_i}{\psi_{i+1} - \psi_i} \right| \leq 1 \quad 
\text{(when } \psi_{i+1} \ne \psi_i \text{)}
\end{aligned}
$$
</div>

- **Type:** Calendar
- **Why it matters:** Limits how quickly the slope of the smile (controlled by \( \rho \psi \)) can change between adjacent maturities. Sudden changes would imply **time arbitrage**, e.g., a butterfly that gains by switching maturities.

---

| Constraint Type | Purpose |
|-----------------|---------|
| **Butterfly**   | Prevent arbitrage **within** a single maturity slice by maintaining convexity of the smile. |
| **Calendar**    | Prevent arbitrage **across** maturities by ensuring smooth growth and transition in variance and skew. |

Together, these constraints form the **arbitrage-free foundation** of the eSSVI volatility surface. Ignoring even one can result in a model that fits market data but allows arbitrage — defeating its practical use in pricing and risk management.

The Constraints were embedded as follows:
```python
def _power_constraints(self, x):
    constraints = []
    phi_his = 0.0
    for T_val in self._T_array:
        rho, eta, gamma = self._compute_parameters(x, T_val)
        phi = eta * (self._theta[T_val] ** -gamma)

        constraints.extend([
            1 - abs(rho) - 1e-5,       # |ρ| < 1
            eta - 1e-5,                # η > 0
            gamma + 0.99,              # γ > -0.99
            0.99 - gamma,              # γ < 0.99
            2 - eta * (1 + abs(rho)),  # η(1 + |ρ|) ≤ 2
            phi - phi_his - 1e-5       # Monotonic φ_t
        ])
        phi_his = phi
    return np.array(constraints)
```
This array would be fed to `minimize` as provided by `SciPy`

## 2. Optimization and Forensic
Now that the structure and intuition are clear, let’s look at what actually broke when I tried to implement this — and how I diagnosed and fixed each issue.


### What Kind of Optimization Problem Is eSSVI Calibration?

Calibrating eSSVI isn’t just curve fitting — it creates a **nonlinear, constrained, nonconvex optimization problem**. Let’s briefly unpack what each of these terms means in this context.

---

#### 1. Nonlinear

- The eSSVI formula involves **products, powers, and square roots**:
<div style="overflow-x: auto;">
  
  $$
  \begin{aligned}
  w(k, t) = \frac{\theta_t}{2} \left(1 + \rho_t \varphi_t k + \sqrt{(\varphi_t k + \rho_t)^2 + 1 - \rho_t^2} \right)
  \end{aligned}
  $$
- Calibration minimizes the difference between **model and market total variances**, making the objective **nonlinear** in parameters. Both the model and loss function are nonlinear.
<\div>
---

#### 2. Constrained

- **Butterfly constraints:** Ensure the smile is convex (e.g., $\eta(1 + |\rho|) < 2 $).
- **Calendar constraints:** Enforce increasing ATM variance and smooth skew behavior across maturities.
- **Bounds:** Parameters like $-1 < \rho_t < 1$, $\varphi_t > 0$ must be enforced. Optimization must respect both hard bounds and arbitrage-free inequalities.

---

#### 3. Nonconvex

- Due to nonlinear interactions and curved constraint regions, the **objective landscape has multiple minima**.
- Optimizers may get stuck depending on initialization. This is a nonconvex problem — no guarantee of finding the global optimum.

---

#### Why It’s Hard in Practice

- **Nonlinearity** makes gradients unpredictable and sensitive.
- **Constraints** limit the search space — you’re optimizing inside a narrow legal region.
- **Nonconvexity** means many traps: local minima that aren’t good enough.
- **Solver issues:** Common errors (like SLSQP’s *exit mode 8*) often occur near boundary violations or in flat regions.

---
### How to solve such problem?

Since most real-world nonlinear, constrained, and nonconvex optimization problems do not have closed-form solutions, **eSSVI calibration is no exception**. Its objective function — often based on squared errors between model and market implied variances — is nonlinear and behaves irregularly across parameter space.

This necessitates the use of **iterative numerical optimization methods** to approximate a feasible and near-optimal solution. Attempts to convert eSSVI into a convex or closed-form compatible formulation are generally impractical due to the model’s structural complexity.

Like any other optimization problem, solving eSSVI calibration involves two key steps:
- **Finding a feasible region** — a set of parameter values that satisfy all arbitrage constraints.
- **Optimizing within that region** — identifying parameter values that best minimize the calibration error.

One effective method used for this is **Sequential Least Squares Quadratic Programming (SLSQP)**, which handles both nonlinear objectives and constraints.

#### Mathematical Formulation of SLSQP (Sequential Least Squares Quadratic Programming)

**SLSQP** is an iterative method for solving **nonlinear constrained optimization problems**. It is particularly useful when both the objective function and the constraints are nonlinear.

##### Problem Setup

SLSQP solves problems of the following general form:

$$
\begin{aligned}
\min_{x \in \mathbb{R}^n} &\quad f(x) \\\\
\text{subject to} &\quad g_j(x) = 0, \quad j = 1, \dots, m_e \\\\
&\quad h_k(x) \geq 0, \quad k = 1, \dots, m_i
\end{aligned}
$$

Where:
- $f(x)$ is the **nonlinear objective function** to minimize.
- $g_j(x)$ are **nonlinear equality constraints**.
- $h_k(x)$ are **nonlinear inequality constraints**.

In our scenario, objective funtion looks something like:
```python
def objective_function(self, x):
    sm = 0.0
    for i, expiry in enumerate(self._T):
        for j, strike in enumerate(self._K):
            model_var = self.total_variance(x, i, j)
            iv = self._iv.loc[expiry, strike]
            if np.isnan(iv):
                continue
            market_var = iv ** 2 * expiry
            weight = 1.0 if self._wgttype == 'none' else self.vega[expiry][j]
            sm += weight * (model_var - market_var) ** 2
    return sm
```
---

##### Core Idea

At each iteration $k$, SLSQP solves a **Quadratic Programming (QP)** subproblem that approximates the original problem locally.

#### QP Subproblem: Approximated Objective

A **quadratic model of the Lagrangian** is used: 
$$
\begin{aligned}
\min_{d \in \mathbb{R}^n} \quad \frac{1}{2} d^T B_k d + \nabla f(x_k)^T d
\end{aligned}
$$

Where:
- $d$ is the search direction.
- $B_k$ approximates the **Hessian** of the Lagrangian using quasi-Newton updates (e.g., BFGS).
- $\nabla f(x_k)$ is the **gradient** of the objective at current iterate $x_k$.

##### Linearized Constraints

Constraints are linearized around the current point:

$$
\begin{aligned}
\nabla g_j(x_k)^T d + g_j(x_k) &= 0, \quad j = 1, \dots, m_e \\\\
\nabla h_k(x_k)^T d + h_k(x_k) &\geq 0, \quad k = 1, \dots, m_i
\end{aligned}
$$
This converts the nonlinear constraints into a **local linear approximation**, making the QP solvable at each iteration.

##### Update Step

Once the direction $d_k$ is found, the solution is updated: $x_{k+1} = x_k + \alpha_k d_k$

Where $\alpha_k \in (0, 1]$ is chosen via a **line search** to ensure:
- Feasibility is preserved,
- Sufficient reduction in the objective.

##### Multiplier Update

Lagrange multipliers (dual variables) are updated using the QP solution.

---

#### Intuition Behind SLSQP

- SLSQP transforms a complex **nonlinear constrained problem** into a sequence of **simpler quadratic subproblems**.
- Constraints are **locally linearized**, making them easier to handle.
- The **Hessian approximation** ($B_k$) gives curvature information, making it faster than purely gradient-based methods.
- The algorithm balances:
  - **Feasibility** (satisfying constraints),
  - **Optimality** (minimizing the objective),
  - Using **merit functions and line search** to avoid divergence.

Finally, the Optimization loop is implemented in `Calibrate()` as follows:
```python
def calibrate(self):
    if self._type == 'Heston':
        init_val = [0.0, 0.0, 1.0, 0.0]
        cons_func = self._heston_constraints
    elif self._type == 'Power':
        init_val = [1.0, -0.05, 0.0, 0.1, 0.0, 0.14]
        cons_func = self._power_constraints

    res = minimize(
        self.objective_function,
        init_val,
        constraints={'type': 'ineq', 'fun': self.cons},
        bounds=self.param_bounds,
        method='SLSQP',
        options={'disp': True, 'maxiter': 220, 'ftol': 1e-8},
        callback=self.callback
    )
    self._x = res.x
    return res.x
    ```
### Forensics

Once the eSSVI parametrization was in place, the real challenge began — **making it fit actual market data**.  
On paper, the model is elegant: a smooth, arbitrage-free surface defined by just a few interpretable parameters.  
In practice, however, the optimization turned out to be far less forgiving.

Even with careful use of optimizers like **SLSQP** and **Differential Evolution**, the solver frequently terminated in unexpected ways — producing high objective values, runtime warnings, or cryptic exit codes like *4*, *5*, and *8*.  
Each of these failure modes turned out to reveal something deeper: about the **mathematical landscape** of eSSVI, the **fragility of its numerical implementation**, and the **sensitivity of the calibration problem** itself.

The table below summarizes the primary errors I encountered and what they revealed about the model:

| Exit Mode / Error | Root Cause | Insight |
|:------------------|:------------|:---------|
| **8 – Positive Directional Derivative** | Flat or ill-conditioned objective; poor initialization; parameter overflow | The optimizer couldn’t find a descent direction — a symptom of eSSVI’s rugged, non-convex loss surface. |
| **5 – Singular Matrix in LSQ** | Numerical breakdown when $\rho > 1$ caused invalid $\sqrt{1 - \rho^2}$ terms | Even a small domain violation poisons the Hessian, showing how sensitive eSSVI is to boundary conditions. |
| **4 – Constraints incompatible or cannot be satisfied** | Theoretical arbitrage condition $\eta(1 + abs(\rho))>2$ became infeasible | Revealed how eSSVI’s feasible region is non-convex and tightly coupled across parameters. |
| **Runtime Warnings** | Overflow in $\phi k + \rho$ or invalid $\sqrt{\cdot}$ terms | Signaled numerical instability and the need for clamping and tighter parameter bounds. |
| **High Objective Values (~10⁶)** | Arbitrary initialization, exploding terms in $\phi = \eta \theta^{-\gamma}$ | Highlighted poor global convergence and the importance of structured initialization. |
| **Gradient Ineffectiveness** | Autograd provided gradients, but they didn’t improve convergence | Gradient quality couldn’t overcome non-convexity or constraint infeasibility. |

After several rounds of diagnostics, I focused primarily on **Exit Mode 4**, since it pointed to a deeper issue — *the solution itself was infeasible under eSSVI’s theoretical constraints*.  

---
#### `Exit Mode 4`: Constraints Incompatible or Cannot be Satisfied
For me, the easiest way to understand what's happening was to **plot the parameter values as a function of** $t$ (time to expiry), in the form: $y = x \cdot m + c$

<div style="position: relative; width: 100%;height: 90%; padding-top: 50%;">
  <iframe 
    src="https://dashessvi-db4kvxfoh-sidkad2003s-projects.vercel.app/plot" 
    style="position: absolute; top: 0; left: 0; width: 100%; height: 90%;" 
    frameborder="0" 
    allowfullscreen>
  </iframe>
</div>
For me, the easiest way to understand what's happening was to **plot the parameter values as a function of** $t$ (time to expiry), in the form: $y = x \cdot m + c$

---






