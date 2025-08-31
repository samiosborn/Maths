# Itô Formula

> **Learning goals**  
> 1) Understand quadratic variation and the heuristic rules \( (dW_t)^2 = dt \), \( dt\,dW_t = 0 \), \( dt^2 = 0 \).  
> 2) State and use Itô’s formula (chain rule) in 1D and multi-dimension.  
> 3) Apply the product rule / integration-by-parts for semimartingales.  
> 4) Convert between Itô and Stratonovich forms.  
> 5) Work through classic examples (e.g. \(W_t^2\), GBM).

---

## 1. Itô Table

- \( (dW_t)^2 = dt \)
- \( dt\,dW_t = 0 \)
- \( dt^2=0 \)

### Proof
Let \( 0=t_0<t_1 < \dots < t_n =T \) be a partition \( \Pi \) of \( [0,T] \), with mesh
\[
|\Pi| = \max_k \Delta t_k
\]

Reminder: 
- Brownian increments \(\Delta W_k\) are independent, mean \(0\), variance \(\mathbb{E}[(\Delta W_k)^2]=\Delta t_k\)
- For \(Z\sim \mathcal N(0,\sigma^2)\), then
\[
\mathbb{E}[Z^2]=\sigma^2,\qquad \operatorname{Var}(Z^2)=2\sigma^4
\]

#### Quadratic variation: \((dW_t)^2 = dt\)

Consider the quadratic variation sum
\[
Q_\Pi = \sum_{k=0}^{n-1} (\Delta W_k)^2
\]

Then, 
\[
\mathbb{E}[Q_\Pi] \;=\; \sum_k \mathbb{E}[(\Delta W_k)^2] \;=\; \sum_k \Delta t_k \;=\; T
\]

And: 
\[
\operatorname{Var}(Q_\Pi)
= \sum_k \operatorname{Var}((\Delta W_k)^2)
= 2 \sum_k (\Delta t_k)^2
\]


Note:

\[
\sum_k (\Delta t_k)^2
\;\le\; \Big(\max_k \Delta t_k\Big) \sum_k \Delta t_k
= |\Pi|\cdot T
\]

So

\[
\operatorname{Var}(Q_\Pi)
= 2\sum_k (\Delta t_k)^2
\;\le\; 2T|\Pi|
\]

As \(|\Pi|\to 0\), the RHS tends to zero. Thus

\[
\operatorname{Var}(Q_\Pi) \;\to\; 0
\]

Now, convergence in \(L^2\) means: 

\[
\mathbb{E}\!\left[\,|Q_\Pi - T|^2\,\right] \;\to\; 0
\]

But, 

\[
\mathbb{E}[Q_\Pi - T] = 0,
\]

so

\[
\mathbb{E}\!\left[\,|Q_\Pi - T|^2\,\right] 
= \operatorname{Var}(Q_\Pi) \;\to\; 0
\]

Therefore

\[
Q_\Pi \;\to\; T
\quad \text{in } L^2
\]

And since \(L^2\) convergence implies convergence in probability, we also have:

\[
Q_\Pi \;\to\; T
\quad \text{in probability}
\]


So, 
\[
(dW_t)^2 = dt
\]

#### Mixed term: \(dt\,dW_t = 0\)

Let: 
\[
M_\Pi = \sum_{k=0}^{n-1} \Delta t_k\, \Delta W_k
\]

We have \(\mathbb{E}[M_\Pi]=0\) as the mean of a Wiener process is zero. 
Also, 
\[
\mathbb{E}[M_\Pi^2]
= \operatorname{Var}(M_\Pi)
= \sum_k (\Delta t_k)^2 \operatorname{Var}(\Delta W_k)
= \sum_k (\Delta t_k)^2 \cdot \Delta t_k
= \sum_k (\Delta t_k)^3
\]
With: 
\[
\sum_k (\Delta t_k)^3 \;\le\; |\Pi| \sum_k (\Delta t_k)^2
\;\le\; |\Pi|\left(\sum_k \Delta t_k\right)^2
= |\Pi|\,T^2 \;\xrightarrow[|\Pi|\to 0]{}\; 0
\]
It follows that \(M_\Pi \to 0\) in \(L^2\). This implies: 
\[
dt\,dW_t = 0
\]

#### Pure time term: \(dt^2 = 0\)

Finally, consider the purely deterministic sum
\[
T_\Pi := \sum_{k=0}^{n-1} (\Delta t_k)^2.
\]
We have the simple bound
\[
0 \;\le\; T_\Pi \;=\; \sum_k (\Delta t_k)^2 \;\le\; |\Pi| \sum_k \Delta t_k \;=\; |\Pi|\,T \;\xrightarrow[|\Pi|\to 0]{}\; 0
\]
Therefore the second-order time terms vanish in the limit:
\[
dt^2 = 0
\]

## 2. Itô Process
Suppose \(X\) has (locally) finite variation, then \([X]_t=0\) and \([X,W]_t=0\) 

For a scalar Itô process: 

\[
dX_t = \mu(t,X_t)\,dt + \sigma(t,X_t)\,dW_t
\]

Where: 
- \( \mu(t,X_t)\,dt \) is the drift term
- \( \sigma(t,X_t)\,dW_t \) is the diffusion term

With quadratic variation: 

\[
d\langle X\rangle_t = \sigma^2(t,X_t)\,dt
\]

---

## 3. Itô’s Formula (1D, time-dependent)

Let \(X_t\) be an Itô process on \([0,T]\):

\[
dX_t = \mu(t,X_t)\,dt + \sigma(t,X_t)\,dW_t,
\]

and let \(f\in C^{1,2}([0,T]\times\mathbb{R})\) (once continuously differentiable in \(t\), twice in \(x\)). 

Then, 
\[
df(t,X_t) \;=\; f_t(t,X_t)\,dt \;+\; f_x(t,X_t)\,dX_t \;+\; \tfrac12 f_{xx}(t,X_t)\,(dX_t)^2
\]

Using the Itô table,

\[
df(t,X_t) \;=\; \Big(f_t + \mu\,f_x + \tfrac12 \sigma^2 f_{xx}\Big)(t,X_t)\,dt \;+\; \sigma(t,X_t) f_x(t,X_t)\,dW_t
\]

Integral form (from \(0\) to \(T\)):

\[
f(T,X_T)-f(0,X_0)
= \int_0^T\!\Big(f_t+\mu f_x+\tfrac12\sigma^2 f_{xx}\Big)(t,X_t)\,dt + \int_0^T\!\sigma f_x(t,X_t)\,dW_t
\]

---

## 4. Multi-Dimensional Itô’s Formula

Suppose \(X_t\in\mathbb{R}^d\) solves the Itô SDE: 

\[
dX_t = \mu(t,X_t)\,dt + \sigma(t,X_t)\,dW_t,
\]

Where 
- \(\mu:\,[0,T]\times\mathbb{R}^d\to\mathbb{R}^d\)
- \(\sigma:\,[0,T]\times\mathbb{R}^d\to\mathbb{R}^{d\times m}\)
- \(W_t\in\mathbb{R}^m\) is an \(m\)-dimensional Brownian motion

Let \(f\in C^{1,2}([0,T]\times\mathbb{R}^d;\mathbb{R})\)

Define \(a(t,x):=\sigma(t,x)\sigma(t,x)^\top\in\mathbb{R}^{d\times d}\)

Then, 

\[
df(t,X_t) = \Big( f_t + \nabla f^\top \mu + \tfrac12 \mathrm{Tr}\!\big(a\,\nabla^2 f\big) \Big)(t,X_t)\,dt
\;+\; \big(\nabla f(t,X_t)^\top \sigma(t,X_t)\big)\,dW_t
\]

In index-form (Einstein summation):

\[
df = \Big(f_t + \mu^i f_{x_i} + \tfrac12 a_{ij} f_{x_i x_j}\Big)dt
\;+\; f_{x_i}\,\sigma^{i}_k\,dW^k.
\]

---

## 5. Product Rule / Integration by Parts

#### Definition: Semimartingale

A stochastic process \(X = (X_t)_{t \ge 0}\) on a filtered probability space \((\Omega, \mathcal{F}, (\mathcal{F}_t), \mathbb{P})\) is called a semimartingale if it can be written as: 

\[
X_t = M_t + A_t
\]

where:
- \(M_t\) is a local martingale
- \(A_t\) is an adapted process of finite variation

Equivalently, semimartingales are the largest class of processes for which the Itô integral can be consistently defined.

### Product Rule

For two semimartingales \(X,Y\):

\[
d(X_t Y_t) = X_t\,dY_t + Y_t\,dX_t + d\langle X,Y\rangle_t
\]

If \(X\) and \(Y\) are Itô processes with diffusions \(\sigma^X,\sigma^Y\) against the same \(m\)-dimensional Brownian motion, then we have the differential (w.r.t. \( t \) ) of the quadratic variation process: 

\[
d\langle X,Y\rangle_t = \sum_{j=1}^m \sigma^X_j(t)\,\sigma^Y_j(t)\,dt
= \big(\sigma^X(t)\big)^\top \sigma^Y(t)\,dt
\]

In particular, for scalar Brownian motion,

\[
d(W_t^2) = 2W_t\,dW_t + dt
\]

---

## 6. Proof Sketch (1D case)

1) Taylor expansion (second order in space, first in time):

\[
f(t+\Delta t, X_{t+\Delta t})
\approx f + f_t\,\Delta t + f_x\,\Delta X + \tfrac12 f_{xx}\,(\Delta X)^2.
\]

2) Order of terms under Brownian scaling:  
\(\Delta X = \mu\,\Delta t + \sigma\,\Delta W\) with \(\Delta W = O(\sqrt{\Delta t})\)
Hence \((\Delta X)^2 = \sigma^2(\Delta W)^2 + o(\Delta t) = \sigma^2 \Delta t + o(\Delta t)\)

3) Sum over a partition, and take limits in probability / \(L^2\), using that sums of \(o(\Delta t)\) vanish and martingale terms converge to stochastic integrals. The surviving drift terms yield \(\tfrac12\sigma^2 f_{xx}\,dt\), producing the formula.

---

## 7. Worked Examples

### 7.1 \(f(x)=x^2\) with \(X_t=W_t\)

Itô with \(f_x=2x,\, f_{xx}=2\):

\[
d(W_t^2) = 2W_t\,dW_t + dt
\quad\Longrightarrow\quad
\int_0^T W_t\,dW_t = \tfrac12\big(W_T^2 - T\big)
\]

### 7.2 Exponential Martingale

Let \((W_t)_{t\ge0}\) be a standard Brownian motion on a filtered probability space
\((\Omega,\mathcal{F},(\mathcal{F}_t)_{t\ge0},\mathbb{P})\)

Fix \(\lambda\in\mathbb{R}\) and define
\[
X_t \;=\; \exp\!\Big(\lambda W_t - \tfrac12 \lambda^2 t\Big), \qquad t\ge 0
\]

Consider \(f(t,w)=\exp(\lambda w - \tfrac12\lambda^2 t)\)
Then
\[
\partial_t f = -\tfrac12\lambda^2 f, \qquad
\partial_w f = \lambda f, \qquad
\partial_{ww} f = \lambda^2 f
\]
By Itô’s formula for \(X_t=f(t,W_t)\)
\[
dX_t
= \partial_t f\,dt + \partial_w f\,dW_t + \tfrac12 \partial_{ww} f\,dt
= \big(-\tfrac12\lambda^2 f + \tfrac12\lambda^2 f\big)\,dt + \lambda f\,dW_t
= \lambda X_t\,dW_t.
\]
So \(X_t\) solves the SDE
\[
dX_t = \lambda X_t\,dW_t, \qquad X_0 = 1.
\]

The stochastic integral \(\int_0^t \lambda X_s\,dW_s\) is a (true) martingale provided
\(\mathbb{E}\int_0^T (\lambda X_s)^2\,ds < \infty\) for each \(T\).
Compute
\[
\mathbb{E}[X_s^2]
= \mathbb{E}\Big[\exp\big(2\lambda W_s - \lambda^2 s\big)\Big]
= \exp\Big(\tfrac12(2\lambda)^2 s\Big)\,\exp(-\lambda^2 s)
= e^{\lambda^2 s},
\]
since \(W_s\sim \mathcal{N}(0,s)\).
Hence
\[
\mathbb{E}\!\int_0^T (\lambda X_s)^2\,ds
= \lambda^2 \int_0^T \mathbb{E}[X_s^2]\,ds
= \lambda^2 \int_0^T e^{\lambda^2 s}\,ds
< \infty.
\]
Therefore \(X_t = X_0 + \int_0^t \lambda X_s\,dW_s\) is a square-integrable martingale.

Because \(X_t\) is a martingale with \(X_0=1\),
\[
\mathbb{E}[X_t] = \mathbb{E}[X_0] = 1 \quad \text{for all } t\ge 0.
\]
(Alternatively, compute directly using \(W_t\sim \mathcal{N}(0,t)\):
\(\mathbb{E}[e^{\lambda W_t}] = e^{\frac12\lambda^2 t}\), so
\(\mathbb{E}[X_t]=e^{\frac12\lambda^2 t} \cdot e^{-\frac12\lambda^2 t}=1\).)

Therefore, 
\(X_t=\exp(\lambda W_t - \tfrac12\lambda^2 t)\) satisfies \(dX_t=\lambda X_t\,dW_t\) and is a martingale with \(\mathbb{E}[X_t]=1\) for all \(t\ge0\).


Then: 

\[
dX_t = \lambda X_t\,dW_t
\]

So, \(X_t\) is a martingale with unit expectation.

### 7.3 Geometric Brownian Motion (GBM)

Suppose: \(dS_t = \mu S_t\,dt + \sigma S_t\,dW_t\) for \(S_0>0\)

Apply Itô formula to: \(f(x)=\log x\)

\[
d\log S_t = \frac{1}{S_t}dS_t - \tfrac12 \frac{1}{S_t^2}(dS_t)^2
= \Big(\mu - \tfrac12\sigma^2\Big)dt + \sigma\,dW_t
\]

Integrating,

\[
S_t = S_0 \exp\!\Big( \big(\mu-\tfrac12\sigma^2\big)t + \sigma W_t \Big)
\]

### 7.4 Ornstein–Uhlenbeck (Identity)

For \(dX_t = \kappa(\theta - X_t)\,dt + \sigma\,dW_t\) 
Itô on \(f(x)=x^2\) gives: 

\[
d(X_t^2) = 2X_t\,dX_t + \sigma^2 dt
= 2\kappa X_t(\theta - X_t)dt + 2\sigma X_t\,dW_t + \sigma^2 dt
\]

Useful for moment calculations.

---

## 8. Itô vs Stratonovich

The Stratonovich integral \(\int X_t \circ dW_t\) obeys the classical chain rule. 

Relation to Itô:
\[
\int_0^T X_t \circ dW_t \;=\; \int_0^T X_t\, dW_t \;+\; \tfrac12\,\langle X,W\rangle_T
\]

### Drift conversion (1D)

Itô SDE:

\[
dX_t = b_{\text{Itô}}(X_t)\,dt + \sigma(X_t)\,dW_t
\]

Equivalent Stratonovich form:

\[
dX_t = b_{\circ}(X_t)\,dt + \sigma(X_t)\circ dW_t
\]

With

\[
b_{\circ}(x) = b_{\text{Itô}}(x) - \tfrac12\,\sigma(x)\,\sigma'(x)
\]

### Drift conversion (multi-D)

If \(dX^i_t = b^i\,dt + \sum_{j} \sigma^{i}_j\,dW^j_t\) (Itô), then Stratonovich drift is

\[
\boxed{ \; b^i_{\circ}(x) = b^i(x) - \tfrac12 \sum_{k,j} \sigma^{k}_j(x)\,\partial_{x_k}\sigma^{i}_j(x). \;}
\]

---

## 9. Common Identities & Tools

- **Generator** \( \mathcal{L} \) of \(X_t\):  
  \[
  \mathcal{L}f = f_t + \mu\cdot\nabla f + \tfrac12 \mathrm{Tr}(a\,\nabla^2 f).
  \]
  If \(f\) is time-independent,  
  \[
  \mathcal{L}f = \mu\cdot\nabla f + \tfrac12 \mathrm{Tr}(a\,\nabla^2 f).
  \]

- **Itô for vector-valued \(f\)**: apply component-wise.  

- **Martingale test**: If \(f\) solves \(f_t + \mathcal{L}f = 0\), then \(f(t,X_t)\) is a local martingale.

---

## 10. Exercises (quick)

1) Use Itô to show \(M_t:=\exp(\lambda W_t - \tfrac12\lambda^2 t)\) is a martingale.  
2) For GBM, apply Itô to \(f(x)=x^p\) and compute \(d(S_t^p)\).  
3) Let \(X_t\) solve \(dX_t = \mu\,dt + \sigma\,dW_t\). Apply Itô to \(f(t,x)=e^{\alpha t}x^2\) and find \(\alpha\) making \(e^{\alpha t}X_t^2 - \int_0^t c\,ds\) a martingale (determine \(c\)).

---

## 11. Summary

- Itô’s formula is the stochastic chain rule accounting for quadratic variation.  
- In 1D:  
  \[
  df = (f_t + \mu f_x + \tfrac12\sigma^2 f_{xx})dt + \sigma f_x dW.
  \]
- In multi-D: replace \( \sigma^2 f_{xx} \) by \( \mathrm{Tr}(a\,\nabla^2 f) \) with \(a=\sigma\sigma^\top\), and the noise term by \( \nabla f^\top \sigma\, dW\).  
- Product rule includes the quadratic covariation term.  
- Stratonovich obeys classical calculus; convert drifts via the \(\tfrac12\) correction.
