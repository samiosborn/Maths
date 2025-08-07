# Kalman Filter

## Statistical Foundations

### Multivariate Gaussian Random Variable 

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
\mathbb{E}\left[ (X_i - \mu_i)(X_j - \mu_j) \right] = \Sigma_{ij} = \mathrm{Cov} (X_i, Z_j)
\]

\[
\mathbb{E}\left[ (X - \mu)(X - \mu)^\top \right] = \Sigma = \mathrm{Var} (X)
\]

---

### Joint Gaussian Random Variables  

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

#### Block Matrix Inversion Formula

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

Schur complements: 
- The Schur complement of \( D \) in \( M \) is:

\[
S = A - B D^{-1} C
\]

- The Schur complement of \( A \) in \( M \) is:

\[
S = D - C A^{-1} B
\]

---

### Precision Matrix  

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

### PDF of \( X \) in terms of \( \Gamma \)

Given: \( \Sigma_{XX} = \Gamma_{XX}^{-1} \)

\[
p(x; \mu, \Gamma) = \frac{1}{(2 \pi)^{n/2} |\Gamma_{XX}|^{-1/2}} \exp\left(-\frac{1}{2} (x-\mu)^\top \Gamma_{XX} (x-\mu)\right)
\]

### Joint PDF of \( X, Y \) in terms of \( \Gamma \)

Given: \( |\Sigma| = |\Gamma|^{-1} \)

\[
p(x, y; \mu, \Gamma) = \frac{1}{(2 \pi)^{n/2} |\Gamma|^{-1/2}} \exp\left(-\frac{1}{2} (x-\mu)^\top \Gamma (x-\mu)\right)
\]

---

### Conditional Distribution of Joint Gaussians

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

This is the canonical form of a univariate Gaussian density with \( X \mid Y \sim \mathcal{N}\left( \mu_{X \mid Y}, \Sigma_{X \mid Y} \right) \):

- Mean: \( \mu_{X \mid Y} = \mu_X + \Sigma_{XY} \Sigma_{YY}^{-1} (y - \mu_Y)\)
- Precision (inverse covariance): \( \Gamma_{XX} \)
- Variance: \(  
\Sigma_{X \mid Y} = \Gamma_{XX}^{-1}
= \Sigma_{XX} - \Sigma_{XY} \Sigma_{YY}^{-1} \Sigma_{YX}
\)

### Intuition from Conditioning Joint Gaussians

The above shows that conditioning a joint Gaussian results in a linear update to the mean and a reduction in the uncertainty (covariance), which is the core idea behind the Kalman filter update step. 

## State-Space Modelling

Time Series analysis uses State-Space Modelling to describe both the underlying latent process,  
\( X_{0:T} := \{x_0, x_1, \dots, x_T\} \), and the (noisy) observations of that underlying process,  
\( Y_{0:T} := \{y_0, y_1, \dots, y_T\} \).

Let:
- \( x_t \in \mathbb{R}^n \): hidden (latent) state at time \( t \)
- \( y_t \in \mathbb{R}^m \): observed measurement at time \( t \)
- \( A_t \in \mathbb{R}^{n \times n} \): state-transition matrix at time \( t \)
- \( H_t \in \mathbb{R}^{m \times n} \): observation matrix at time \( t \)
- \( w_t \sim \mathcal{N}(0, Q_t) \): process (motion) noise
- \( v_t \sim \mathcal{N}(0, R_t) \): observation noise

---

### System (Process) Model

\[
x_{t+1} = f_t(x_t) + w_t, \quad w_t \sim \mathcal{N}(0, Q_t)
\]

- \( f_t: \mathbb{R}^n \to \mathbb{R}^n \): state transition function  
- \( Q_t \): process noise covariance at time \( t \)

---

### Observation (Channel) Model

\[
y_t = h_t(x_t) + v_t, \quad v_t \sim \mathcal{N}(0, R_t)
\]

- \( h_t: \mathbb{R}^n \to \mathbb{R}^m \): observation function  
- \( R_t \): observation noise covariance at time \( t \)

---

### Recursive Joint Distribution

Instead of directly modelling the full joint distribution:

\[
p(x_0, x_1, \dots, x_T, y_0, y_1, \dots, y_T)
\]

We use the Markov structure to factorise it recursively:

\[
p(X_{0:T}, Y_{0:T}) = p(x_0) \prod_{t=1}^T p(x_t \mid x_{t-1}) \prod_{t=0}^T p(y_t \mid x_t)
\]

This follows from:
- The first-order Markov property of the latent states \( x_t \)
- Conditional independence of observations \( y_t \) given the current state \( x_t \)

---

### Linear-Gaussian State-Space Model

Assumptions: 
- The state-transition function is directly linear (at time t): \( f_t(x_t) = A_t x_t \)
- The observation function is a directly linear (at time t): \( h_t(x_t) = H_t x_t \)

With:
- \( A_t \in \mathbb{R}^{n \times n} \): state-transition matrix at time \( t \)
- \( H_t \in \mathbb{R}^{m \times n} \): observation matrix at time \( t \)

Process Model: 
\[
x_{t+1} = A_t x_t + w_t, \quad w_t \sim \mathcal{N}(0, Q_t)
\]

Observation Model: 
\[
y_t = H_t x_t + v_t, \quad v_t \sim \mathcal{N}(0, R_t)
\]

This is a time-varying linear-Gaussian state-space model, which is the setting for the standard Kalman Filter.

## Recursive Bayesian Estimation

Given the state-space model, we now aim to recursively estimate the hidden state \( x_t \) from the sequence of noisy observations \( y_{0:t} \). That is, we wish to compute:

\[
p(x_t \mid y_{0:t})
\]

Define: 
- \( \mathbb{E} [X_t \mid Y_{0:t}] = \hat{\mu}_{t} \)
- \( \mathrm{Var} [X_t \mid Y_{0:t}] = P_{t} \)

This posterior distribution is updated over time using two recursive steps: prediction, and then update. 

### Prediction Step

Before we observe \( y_t \) we want to predict forward one step the prior distribution, over the hidden state at time \( t \), using all past observations up to time \( t-1 \):
\[ 
p(x_t \mid y_{0:t-1}) 
\]

Define: 
- \( \mathbb{E} [X_t \mid Y_{0:t-1}] = \hat{\mu}_{t}^- \)
- \( \mathrm{Var} [X_t \mid Y_{0:t-1}] = P_{t}^- \)

By marginalising the joint distribution \( p(x_t, x_{t-1} \mid y_{0:t-1}) \) over the possible values of the previous hidden state \( x_{t-1} \) we have:

\[
p(x_t \mid y_{0:t-1}) = \int p(x_t, x_{t-1} \mid y_{0:t-1}) \, dx_{t-1}
\]

By applying the chain rule of conditional probabilities:

\[
p(x_t, x_{t-1} \mid y_{0:t-1}) = p(x_t \mid x_{t-1}, y_{0:t-1}) \, p(x_{t-1} \mid y_{0:t-1})
\]

We now use the Markov assumption that the state \( x_t \) depends only on the previous state \( x_{t-1} \), and not on earlier observations. So:

\[
p(x_t \mid x_{t-1}, y_{0:t-1}) = p(x_t \mid x_{t-1})
\]

Substituting this back gives the prediction equation integral:

\[
p(x_t \mid y_{0:t-1}) = \int p(x_t \mid x_{t-1}) \, p(x_{t-1} \mid y_{0:t-1}) \, dx_{t-1}
\]

Using the linear-Gaussian state-space model:
\[
x_t = A_{t-1} x_{t-1} + w_{t-1}, \quad w_{t-1} \sim \mathcal{N}(0, Q_{t-1}) \\
y_t = H_t x_t + v_t, \quad v_t \sim \mathcal{N}(0, R_t)
\]

We can assume the posterior at time \( t-1 \) is:
\[
X_{t-1} \mid Y_{0:t-1} \sim \mathcal{N}(\hat{\mu}_{t-1}, P_{t-1})
\]

And the prior distribution at time \( t \) is Gaussian:
\[
X_t \mid Y_{0:t-1} \sim \mathcal{N}(\hat{\mu}_t^-, P_t^-)
\]

We have the prediction step mean: 
\[
\hat{\mu}_{t}^- = \mathbb{E} [X_{t} \mid Y_{0:t-1}] =  \mathbb{E} [X_{t-1} + A_{t-1} X_{t-1} + W_{t-1} \mid Y_{0:t-1}] = A_{t-1} \hat{\mu}_{t-1} 
\]

For the prediction step covariance: 
\[
P_t^- = \mathrm{Var}[X_t \mid Y_{0:t-1}] = \mathbb{E} \left[ (X_t - \hat{\mu}_t^-) (X_t - \hat{\mu}_t^-)^\top \mid Y_{0:t-1} \right]
\]

Expanding: 
\[
P_t^- = \mathbb{E} \left[ (A_{t-1} X_{t-1} + W_{t-1} - A_{t-1} \hat{\mu}_{t-1}) (A_{t-1} X_{t-1} + W_{t-1} - A_{t-1} \hat{\mu}_{t-1})^\top \mid Y_{0:t-1} \right]
\]

Using conditional independence: 
\[
P_t^- =  A_{t-1} \mathbb{E} \left[ (X_{t-1} - \hat{\mu}_{t-1}) (X_{t-1} - \hat{\mu}_{t-1})^\top \mid Y_{0:t-1} \right]  A_{t-1}^\top + \mathrm{Var} [ W_{t-1} ]
\]

Therefore:
\[
P_t^- = A_{t-1} P_{t-1} A_{t-1}^\top + Q_{t-1}
\]

In summary: 
\[
X_t \mid Y_{0:t-1} \sim \mathcal{N}( A_{t-1} \hat{\mu}_{t-1}, \quad A_{t-1} P_{t-1} A_{t-1}^\top + Q_{t-1} )
\]

---

### Update Step

After making a prediction for the hidden state \( x_t \), we observe \( y_t \). This allows us to update our prior prediction into a posterior estimate. \\

Using Bayes’ Theorem and the conditional independence assumption, we have:

\[
p(x_t \mid y_{0:t}) = \frac{p(y_t \mid x_t) \, p(x_t \mid y_{0:t-1})}{p(y_t \mid y_{0:t-1})}
\]

Prediction state distribution: \( p(x_t \mid y_{0:t-1}) \)
- In this setting, we know from above step: \( X_t \mid Y_{0:t-1} \sim \mathcal{N}( A_{t-1} \hat{\mu}_{t-1}, \quad A_{t-1} P_{t-1} A_{t-1}^\top + Q_{t-1} ) \)

Likelihood: \( p(y_t \mid x_t) \)
- Based on the observation model: \( Y_t \mid X_t = x_t \sim \mathcal{N} ( H_t x_t, R_t ) \)

Evidence (normalising constant) integral: 
- Marginalising integral: \( p(y_t \mid y_{0:t-1}) = \int p(y_t \mid x_t) \, p(x_t \mid y_{0:t-1}) \, dx_t \)

---

### Linear-Gaussian Case (Kalman Filter)

If both the state transition and observation models are linear, and all noise terms are Gaussian, then all distributions remain Gaussian at each time step.

Specifically, we model:

- \( X_t \mid Y_{0:t-1} \sim \mathcal{N}(\hat{\mu}_t^-, P_t^-) \): prior from the prediction step
- \( Y_t \mid X_t = x_t \sim \mathcal{N}(H_t x_t, R_t) \): likelihood from observation model

The joint distribution \( (X_t, Y_t) \) of the two Gaussians is: 

\[
\begin{bmatrix}
X_t \\
Y_t
\end{bmatrix}
\sim \mathcal{N}\left(
\begin{bmatrix}
\mathbb{E}[X_t] \\
\mathbb{E}[Y_t]
\end{bmatrix},
\begin{bmatrix}
\mathrm{Var} [X_t] & \mathrm{Cov} [X_t, Y_t] \\
\mathrm{Cov} [Y_t, X_t] & \mathrm{Var} [Y_t]
\end{bmatrix}
\right)
\]

Where: 
- \( \mathbb{E}[X] = \hat{\mu}_t^-\)
- \( \mathbb{E}[Y] = H_t \mathbb{E} [X_t] = H_t \hat{\mu}_t^- \)
- \( \mathrm{Var} [X_t] = P_t^- \)
- \( \mathrm{Var} [Y_t] = R_t \)
- \( \mathrm{Cov} [X_t, Y_t] = \mathbb{E} [(X_t - \hat{\mu}_t^-)(H_t X_t + V_t - H_t \hat{\mu}_t^-)^\top] = P_t^- H_t^\top \)
- \( \mathrm{Cov} [Y_t, X_t] = \mathrm{Cov} [X_t, Y_t]^\top = H_t P_t^- \)

So: 
\[
\begin{bmatrix}
X_t \\
Y_t
\end{bmatrix}
\sim \mathcal{N}\left(
\begin{bmatrix}
\hat{\mu}_t^- \\
H_t \hat{\mu}_t^-
\end{bmatrix},
\begin{bmatrix}
P_t^- & P_t^- H_t^\top \\
H_t P_t^- & H_t P_t^- H_t^\top + R_t
\end{bmatrix}
\right)
\]

Reminder: Conditioning a joint Gaussian:

\[
X \mid Y = y \sim \mathcal{N}\left(
\mathrm{E}[X] + \Sigma_{XY} \Sigma_{YY}^{-1} (y - \mathrm{E}[Y]),
\Sigma_{XX} - \Sigma_{XY} \Sigma_{YY}^{-1} \Sigma_{YX}
\right)
\]

Then the conditional distribution is:

\[
X_t \mid Y_{0:t} \sim \mathcal{N}(\hat{\mu}_t, P_t)
\]

Posterior Mean:

\[
\hat{\mu}_t = \hat{\mu}_t^- + P_t^- H_t^\top (H_t P_t^- H_t^\top + R_t)^{-1} (y_t - H_t \hat{\mu}_t^-)
\]

Posterior Covariance:

\[
P_t = P_t^- - P_t^- H_t^\top (H_t P_t^- H_t^\top + R_t)^{-1} H_t P_t^- 
\]

#### Kalman Gain

Consider the measurement residual: \( y_t - H_t \hat{\mu}^- \)

Define the Kalman Gain: 
\[
K_t = P_t^- H_t^\top (H_t P_t^- H_t^\top + R_t)^{-1} 
\]

Redefining the Posterior Mean using the Kalman Gain:  
\[
\hat{\mu}_t = \hat{\mu}_t^- + K_t (y_t - H_t \hat{\mu}_t^-)
\]

Redefining the Posterior Covariance using the Kalman Gain: 

\[
P_t = (I - K_t H_t) P_t^-
\]

- \( K_t \) balances trust between the prediction and new observation. 
- If \( R_t \) is large (i.e. noisy observations), \( K_t \to 0 \): trust prediction.
- If \( P_t^- \) is large (i.e. uncertain prediction), \( K_t \to 1 \): trust measurement.

---

### Initialisation

To begin the recursive estimation, we assume a prior distribution over the initial hidden state:

\[
x_0 \sim \mathcal{N}(\hat{\mu}_0, P_0)
\]

This prior feeds into the first prediction step, so the quality of the initial estimate affects how quickly the filter converges.

There are 3 main strategies for choosing the initial prior:


- Informative Prior: For example, a robot which starts at the origin with high confidence:

\[
\hat{\mu}_0 = 0, \quad P_0 = \text{small}
\]

- Uninformative Prior: If we are highly uncertain, use a large covariance:

\[
P_0 = \alpha I, \quad \text{with } \alpha \gg 1
\]

- Steady-State Estimate: Initialise using the theoretical solution to the steady-state situation.

---
 
### Linear Minimum Mean Squared Error (LMMSE) Estimation Problem

The LMMSE estimator is the linear function, \( \hat{x}_t\), that minimises the mean squared error:

\[
\hat{x}_t = \arg\min_{\hat{x}_t \in \mathcal{L}} \mathbb{E}\left[ \| x_t - \hat{x}_t \|^2 \right]
\]

Where \( \mathcal{L} \) is the set of all linear estimators of the form: \( \hat{x}_t = L y_{0:t} + b \) where \(L\) is a matrix and \(b\) is a constant. 

If \( x_t \) and \( y_{0:t} \) are jointly Gaussian, then the LMMSE estimator is the conditional mean:

\[
\hat{x}_t = \mathbb{E}[ x_t \mid y_{0:t} ]
\]

By definiton of the Kalman filter as the conditional mean, it therefore is the LMMSE estimator: 

\[
\hat{x}_t = \hat{\mu}_t = \mathbb{E}[x_t \mid y_{0:t}]
\]

---
