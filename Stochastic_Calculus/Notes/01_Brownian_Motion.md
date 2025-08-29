# Brownian Motion (Wiener Process)

> **Learning Goals:** 
> 1) Build an intuition for Brownian motion as the continuum limit of random walks.  
> 2) Know the definitions and the most important properties.  
> 3) Compute basic moments and the quadratic variation.  
> 4) Run simulations that demonstrate Brownian motion.

---

## 1 Intuition

Consider a symmetric random walk \( \{S_n\} \) with step size \( \Delta W \) at discrete time step \(\Delta t\).

Suppose
\[
\Delta W_k \in \{ +\sqrt{\Delta t},\; -\sqrt{\Delta t} \}, \quad \mathbb P(\Delta W_k = \pm \sqrt{\Delta t}) = \tfrac{1}{2}.
\]

Define the position at time \(t_n = n\,\Delta t\) as
\[
W_{t_n} \;=\; \sum_{k=1}^{n} \Delta W_k .
\]

As \(\Delta t \to 0\) (equivalently \(n \to \infty\) while keeping \(t_n = n\,\Delta t\) fixed), the process \(\{W_{t_n}\}\) converges in distribution to a continuous Wiener process \(W_t\) with independent Gaussian increments:
\[
W_t - W_s \sim \mathcal N(0,\, t-s).
\]

This limiting process \(W_t\) is standard Brownian motion.

---

## 2 Definition

A process \(W_t\) is a Wiener process if:

1. \(W_0 = 0\) almost surely (i.e. \( \mathbb{P}(W_0 = 0) = 1 \) )
2. Independent increments: all increments \(W_{t_1}-W_{t_0},\dots,W_{t_n}-W_{t_{n-1}}\) are independent.
3. Gaussian increments: for all \(s<t\) ,
\[
W_t - W_s \sim \mathcal N(0,\,t-s)
\]
4. Continuous paths: \(t \mapsto W_t\) is continuous almost surely.

Quadratic Variation:
\[
dW_t \sim \mathcal N(0,\,dt), \qquad (dW_t)^2\approx dt.
\]

---

## 3 Properties

### 3.1 Mean and variance
For all \( t \geq 0\):
\[
\mathbb E[W_t]=0,\qquad \mathrm{Var}(W_t)=t
\]
Given: \(W_t = W_t - W_0\sim \mathcal N(0,t)\)

### 3.2 Covariance
For \(s\le t\),
\[
\mathrm{Cov}(W_s,W_t)=\mathbb E[W_s W_t] = s
\]
Hint: Write \(W_t=W_s+(W_t-W_s)\), and recall independent increment of zero mean.

### 3.3 Independent Increments
For any \(h>0\), 

\(W_{t+h}-W_t \sim \mathcal N(0,h)\)

### 3.4 Scaling (self-similarity)
For any \(c>0\),
\[
\{W_{ct}\}_{t\ge 0}\;\stackrel{d}{=}\;\{\sqrt{c}\,W_t\}_{t\ge 0}.
\]

Sketch: Match the finite-dimensional distributions (Gaussian with matching means/covariances).

---

## 4. Quadratic Variation

Let \(0=t_0<\dots < t_n=t \) be a partition. 

The mesh size (or norm) of the partition is defined as: 
\[
|\Pi| = \max_{0 \le k \le n-1} (t_{k+1} - t_k)
\]

Write \(\Delta W_k=W_{t_{k+1}}-W_{t_k}\)

The quadratic variation of \(W\) on \([0,t]\) is
\[
[W]_t = \lim_{\lvert\Pi\rvert\to 0}\ \sum_{k=0}^{n-1} (\Delta W_k)^2
\]

Theorem: For Brownian motion, \([W]_t = t\) almost surely.

Since: \(\mathbb E\big[\sum (\Delta W_k)^2\big]=\sum \mathbb E[(\Delta W_k)^2]=\sum (t_{k+1}-t_k)=t\)

---

## 5. Reflection Principle

For \(a>0\) and \(t>0\):
\[
\mathbb P\!\left(\max_{0\le s\le t} W_s \ge a\right) = 2\,\mathbb P(W_t\ge a) = 2\left(1-\Phi\!\left(\frac{a}{\sqrt{t}}\right)\right)
\]
Where \(\Phi\) is the standard normal CDF.

Proof: 
Let \(M_t := \max_{0\le s\le t} W_s\) and fix \(a>0\).

If a path first crosses \(a\) at time \(\tau\), reflect the segment after \(\tau\) across the horizontal line \(y=a\):
\[
W_s' \;=\;
\begin{cases}
W_s, & s\le \tau,\\[4pt]
2a - W_s, & s>\tau.
\end{cases}
\]

\[
\mathbb{P}(M_t \ge a)
= \mathbb{P}(W_t \ge a) + \mathbb{P}(M_t \ge a,\; W_t < a)
\]

\[
\mathbb{P}(M_t \ge a)
= \mathbb{P}(W_t \ge a) + \mathbb{P}(W_t \ge a)
= 2\,\mathbb{P}(W_t \ge a).
\]

Since \(W_t \sim \mathcal N(0,t)\),
\[
\boxed{\;\mathbb{P}(M_t \ge a) = 2\!\left(1-\Phi\!\left(\tfrac{a}{\sqrt{t}}\right)\right)\;}
\]
where \(\Phi\) is the standard normal CDF.

Intuition: “Crossed \(a\) and ended below” ↔ “end above \(a\)” via reflecting the tail of the path after the first hit. Because Brownian motion forgets the past at hitting times and has symmetric increments, this reflection doesn’t change probabilities, so the two events have the same chance.

---

## 6. Multidimensional Brownian Motion

Define: \(\mathbf W_t=(W_t^{(1)},\dots,W_t^{(d)})\) with independent coordinates, each a standard 1-D Brownian motion. 

Then, \(\mathbf W_t-\mathbf W_s\sim\mathcal N(\mathbf 0,(t-s)I_d)\).

---
