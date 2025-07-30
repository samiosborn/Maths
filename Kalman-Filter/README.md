# Kalman Filter

## Multivariate Gaussian Random Variable 

Let \( X \) represent a multivariate Gaussian random variable.  

The probability density function (PDF) \( p \) is defined as:  

\[
p(x; \mu, \Sigma) = \frac{1}{(2 \pi)^{n/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (x-\mu)^\top \Sigma^{-1} (x-\mu)\right)
\]

From the definition of expectation:  

\[
\mathbb{E}[X_i] = \int x_i \, p(x; \mu, \Sigma) \, dx = \mu_i
\]

\[
\mathbb{E}[X] = \int x \, p(x; \mu, \Sigma) \, dx = \mu
\]

From the definition of covariance:  

\[
\mathbb{E}\left[ (X_i - \mu_i)(X_j - \mu_j) \right] = \Sigma_{ij} = Cov(X_i, Z_j)
\]

\[
\mathbb{E}\left[ (X - \mu)(X - \mu)^\top \right] = \Sigma = Var(X)
\]

---

## Joint Gaussian Random Variables  

For two Gaussian random variables \( X \) and \( Y \):

\[
\begin{bmatrix}
X \\
Y
\end{bmatrix}
\sim \mathcal{N}\left(
\begin{bmatrix}
\mu_X \\
\mu_Y
\end{bmatrix},
\begin{bmatrix}
\Sigma_{XX} & \Sigma_{XY} \\
\Sigma_{YX} & \Sigma_{YY}
\end{bmatrix}
\right)
\]

The joint pdf is:

\[
p\left(
\begin{bmatrix}
x \\
y
\end{bmatrix}
\right)
= \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} 
\exp\left( 
-\frac{1}{2}
\left(
\begin{bmatrix}
x \\
y
\end{bmatrix} - 
\begin{bmatrix}
\mu_X \\
\mu_Y
\end{bmatrix}
\right)^\top
\Sigma^{-1}
\left(
\begin{bmatrix}
x \\
y
\end{bmatrix} -
\begin{bmatrix}
\mu_X \\
\mu_Y
\end{bmatrix}
\right)
\right)
\]

Expectation and covariance definitions:

\[
\begin{aligned}
\mu_X &= \mathbb{E}[X] \\
\mu_Y &= \mathbb{E}[Y] \\
\Sigma_{XX} &= \mathbb{E}\left[ (X - \mu_X)(X - \mu_X)^\top \right] \\
\Sigma_{YY} &= \mathbb{E}\left[ (Y - \mu_Y)(Y - \mu_Y)^\top \right] \\
\Sigma_{XY} &= \mathbb{E}\left[ (X - \mu_X)(Y - \mu_Y)^\top \right] = \Sigma_{YX}^\top \\
\Sigma_{YX} &= \mathbb{E}\left[ (Y - \mu_Y)(X - \mu_X)^\top \right] = \Sigma_{XY}^\top
\end{aligned}
\]

---

### Block Matrix Inversion Formula

Let \( M \) be a square invertible matrix: 

\[
M =
\begin{bmatrix}
A & B \\
C & D
\end{bmatrix}
\]

Then the inverse is:

\[
M^{-1} =
\begin{bmatrix}
A & B \\
C & D
\end{bmatrix}^{-1} =
\begin{bmatrix}
(A - B D^{-1} C)^{-1} & -A^{-1} B (D - C A^{-1} B)^{-1} \\
 -D^{-1} C (A - B D^{-1} C)^{-1} & (D - C A^{-1} B)^{-1}
\end{bmatrix}
\]

---

## Precision Matrix  

Define the precision matrix \( \Gamma \) as the inverse of the covariance matrix \( \Sigma \):  

\[
\Gamma = \Sigma^{-1} = 
\begin{bmatrix}
\Gamma_{XX} & \Gamma_{XY} \\
\Gamma_{YX} & \Gamma_{YY}
\end{bmatrix}
\]

Using the Block Inversion above:

\[
\Sigma_{XX} = \left( \Gamma_{XX} - \Gamma_{XY} \Gamma_{YY}^{-1} \Gamma_{YX} \right)^{-1}
\]

\[
\Sigma_{YY} = \left( \Gamma_{YY} - \Gamma_{YX} \Gamma_{XX}^{-1} \Gamma_{XY} \right)^{-1}
\]

\[
\Sigma_{XY} = -\Gamma_{XX}^{-1} \Gamma_{XY} \left( \Gamma_{YY} - \Gamma_{YX} \Gamma_{XX}^{-1} \Gamma_{XY} \right)^{-1}
\]

\[
\Sigma_{YX} = -\Gamma_{YY}^{-1} \Gamma_{YX} \left( \Gamma_{XX} - \Gamma_{XY} \Gamma_{YY}^{-1} \Gamma_{YX} \right)^{-1}
\]

Similarly for the Covariance matrix: 

\[
\Gamma_{XX} = \left( \Sigma_{XX} - \Sigma_{XY} \Sigma_{YY}^{-1} \Sigma_{YX} \right)^{-1}
\]

\[
\Gamma_{YY} = \left( \Sigma_{YY} - \Sigma_{YX} \Sigma_{XX}^{-1} \Sigma_{XY} \right)^{-1}
\]

\[
\Gamma_{XY} = -\Sigma_{XX}^{-1} \Sigma_{XY} \left( \Sigma_{YY} - \Sigma_{YX} \Sigma_{XX}^{-1} \Sigma_{XY} \right)^{-1}
\]

\[
\Gamma_{YX} = -\Sigma_{YY}^{-1} \Sigma_{YX} \left( \Sigma_{XX} - \Sigma_{XY} \Sigma_{YY}^{-1} \Sigma_{YX} \right)^{-1}
\]

---

## PDF of \( X \) in terms of \( \Gamma \)

Given: \( |\Sigma_{XX}| = |\Gamma_{XX}|^{-1} \)

\[
p(x; \mu, \Gamma) = \frac{1}{(2 \pi)^{n/2} |\Gamma_{XX}|^{-1/2}} \exp\left(-\frac{1}{2} (x-\mu)^\top \Gamma_{XX} (x-\mu)\right)
\]

## Joint PDF of \( X, Y \) in terms of \( \Gamma \)

Given: \( |\Sigma|^{1/2} = |\Gamma|^{-1/2} \)

\[
p(x, y; \mu, \Gamma) = \frac{1}{(2 \pi)^{n/2} |\Gamma|^{-1/2}} \exp\left(-\frac{1}{2} (x-\mu)^\top \Gamma (x-\mu)\right)
\]

---

## Conditional Distribution of Joint Gaussians

Let the joint distribution of \( X \in \mathbb{R}^n \) and \( Y \in \mathbb{R}^m \) be:

\[
\begin{bmatrix}
X \\
Y
\end{bmatrix}
\sim \mathcal{N}\left(
\begin{bmatrix}
\mu_X \\
\mu_Y
\end{bmatrix},
\begin{bmatrix}
\Sigma_{XX} & \Sigma_{XY} \\
\Sigma_{YX} & \Sigma_{YY}
\end{bmatrix}
\right)
\]

From the definition of conditional probability, we have the pdf of the conditional distribution: 
\[
p(x; y) =  \frac{p(x, y)}{p(y)} 
\]

Focusing on the exponential: 
\[
p(x; y) \propto 
\exp\left(
-\frac{1}{2}
\begin{bmatrix}
x - \mu_X \\
y - \mu_Y
\end{bmatrix}^\top
\begin{bmatrix}
\Gamma_{XX} & \Gamma_{XY} \\
\Gamma_{YX} & \Gamma_{YY}
\end{bmatrix}
\begin{bmatrix}
x - \mu_X \\
y - \mu_Y
\end{bmatrix}
+\frac{1}{2} (y-\mu_Y)^\top \Gamma_{YY} (y-\mu_Y)
\right)
\]

Note how the above last term is just in terms of \( y \) which is a constant and we can disregard. 

Let \( u = x - \mu_X \), \( v = y - \mu_Y \). Then:
\[
p(x; y) \propto 
\exp\left(
-\frac{1}{2}
\begin{bmatrix}
u \\
v
\end{bmatrix}^\top
\begin{bmatrix}
\Gamma_{XX} & \Gamma_{XY} \\
\Gamma_{YX} & \Gamma_{YY}
\end{bmatrix}
\begin{bmatrix}
u \\
v
\end{bmatrix}
\right)
\]

Taking the log and expanding: 
\[
\log p(x ; y) \propto -\frac{1}{2} \left(
u^\top \Gamma_{XX} u + u^\top \Gamma_{XY} v + v^\top \Gamma_{YX} u + v^\top \Gamma_{YY} v
\right)
\]

Because \( \Gamma \) is symmetric (as the inverse of a covariance matrix), we know \( \Gamma_{YX} = \Gamma_{XY} \), so the cross terms combine:

\[
\log p(x ; y) \propto -\frac{1}{2}
\left(
u^\top \Gamma_{XX} u + 2 u^\top \Gamma_{XY} v + v^\top \Gamma_{YY} v
\right)
\]

We now aim to complete the square in \( u \). Let: \( A = \Gamma_{XX}, B = \Gamma_{XY} \)

\[
\log p(x ; y) \propto -\frac{1}{2} \left( u^\top A u + 2 u^\top B v + v^\top \Gamma_{YY} v \right)
\]

Observe that:

\[
\left(u + A^{-1} B v\right)^\top A \left(u + A^{-1} B v\right)  = u^\top A u + 2 u^\top B v + v^\top B^\top A^{-1} B v
\]

So we have:

\[
\log p(x ; y) \propto -\frac{1}{2} \left[
\left(u + A^{-1} B v\right)^\top A \left(u + A^{-1} B v\right) - v^\top B^\top A^{-1} B v + v^\top \Gamma_{YY} v
\right]
\]

Substituting back gives:

\[
\log p(x ; y) \propto -\frac{1}{2} \left[
\left(u + \Gamma_{XX}^{-1} \Gamma_{XY} v\right)^\top \Gamma_{XX} \left(u + \Gamma_{XX}^{-1} \Gamma_{XY} v\right) - v^\top \Gamma_{YX} \Gamma_{XX}^{-1} \Gamma_{XY} v + v^\top \Gamma_{YY} v
\right]
\]

Dropping the terms independent of \( x \) (i.e. dropping terms with just \( v \)) the conditional density becomes:

\[
\log p(x ; y) \propto -\frac{1}{2} \left[
\left(u + \Gamma_{XX}^{-1} \Gamma_{XY} v\right)^\top \Gamma_{XX} \left(u + \Gamma_{XX}^{-1} \Gamma_{XY} v\right) 
\right]
\]

If we let:  
\[
\mu_{X \mid Y} = \mu_X - \Gamma_{XX}^{-1} \Gamma_{XY} v 
= \mu_X + \Sigma_{XY} \Sigma_{YY}^{-1} (y - \mu_Y)
\]

since \( \Gamma_{XX}^{-1} \Gamma_{XY} = - \Sigma_{XY} \Sigma_{YY}^{-1} \), then the quadratic expression becomes:

\[
p(x ; y) \propto
\exp\left(- \frac{1}{2}
(x - \mu_{X \mid Y})^\top \Gamma_{XX} (x - \mu_{X \mid Y})
\right)
\]

This is the canonical form of a Gaussian density with \( X \mid Y \sim \mathcal{N}\left( \mu_{X \mid Y}, \Sigma_{X \mid Y} \right) \):

- Mean: \( \mu_{X \mid Y} = \mu_X + \Sigma_{XY} \Sigma_{YY}^{-1} (y - \mu_Y)\)
- Precision (inverse covariance): \( \Gamma_{XX} \)
- Variance: \(  
\Sigma_{X \mid Y} = \Gamma_{XX}^{-1}
= \Sigma_{XX} - \Sigma_{XY} \Sigma_{YY}^{-1} \Sigma_{YX}
\)

## Intuition

The above shows that conditioning a joint Gaussian results in a linear update to the mean and a reduction in the uncertainty (covariance), which is the core idea behind the Kalman filter update step. 

