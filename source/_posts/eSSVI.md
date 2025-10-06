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

$$
\begin{aligned}
dS_t &= \mu S_t\,dt + \sqrt{V_t}\,S_t\,dW_t^S \\
dV_t &= \kappa(\theta - V_t)\,dt + \xi\sqrt{V_t}\,dW_t^V \\
dW_t^S\,dW_t^V &= \rho\,dt
\end{aligned}
$$

### Explanation of Terms

| Symbol | Meaning |
|:--------|:--------|
| \(S_t\) | Underlying asset price |
| \(V_t\) | Instantaneous (stochastic) variance |
| \(\mu\) | Drift of the asset price |
| \(\kappa\) | Speed of mean reversion of variance |
| \(\theta\) | Long-term mean of variance |
| \(\xi\) | Volatility of volatility |
| \(\rho\) | Correlation between the asset and variance processes |
| \(dW_t^S, dW_t^V\) | Brownian motions driving randomness |

### Key Idea

The asset price \(S_t\) evolves under **random shocks**, influenced by both its own dynamics and a **stochastic variance** \(V_t\).  
This dual randomness captures features observed in real markets — most notably **volatility smiles and skews** in implied volatility surfaces derived from option prices.




