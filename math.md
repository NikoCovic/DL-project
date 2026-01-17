# Muon Edge of Stability

## Muon Definition

Suppose we have a Neural Network $f_\theta$ which contains $L$ rectangular layers and a total of $p$ parameters. Let $\mathcal{L}: \mathbb{R}^p \rightarrow \mathbb{R}$ be some loss function. The parameters are defined as a set of the rectangular matrices (assume only those exist for simplicity, the rest might be optimized with a different optimizer). So, $\theta = \{W_1, \dots, W_L\}$

First, the gradient at time $t$ is equal to 

$$
G_t = \nabla_\theta\mathcal{L}(\theta_{t-1})
$$

Let $G_t^{(l)}$ be the part of the gradient corresponding to $W_{t-1}^{(l)}$ at time $t$ (both are matrices). The vanilla Muon update at time $t$ for each layer $1 \le l \le L$ is then:

$$
\begin{aligned}

M_t^{(l)} &= \beta M_{t-1}^{(l)} + G_t^{(l)} \\

O_t^{(l)} &= \text{orthogonalize}(M_t^{(l)}) \\

W_{t}^{(l)} &= W_{t-1}^{(l)} - \eta O_t^{(l)}

\end{aligned}
$$

## Preconditioner

The aim here is to re-write the per-layer Muon update as:

$$
w_t^{(l)} = w_{t-1}^{(l)} - \eta D_t^{(l)-1}m_t^{(l)}
$$

where $w_t^{(l)} = \text{vec}(W_t^{(l)})$, and the same holds for the other parameters and $D_t^{(l)}$ is some matrix.

First, lets write it without the vectorization, i.e. $W_t^{(l)} = W_{t-1}^{(l)} - \eta P_t^{(l)-1}M_t^{(l)}$. It is a known fact that orthogonalization can be done by setting all the singular values of the matrix to 1. Therefore, if $M_t^{(l)} = U \Sigma V^T$ is the SVD of $M_t^{(l)}$, then $O_t^{(l)} = UV^T$. I will show that $O_t^{(l)} = (M_t^{(l)}M_t^{(l)T})^{-1/2}M_t^{(l)}$ (I'll drop the $t$ and $(l)$ for clarity):

$$
\begin{aligned}

(MM^T)^{-1/2}M &= (U\Sigma V^T V \Sigma U^T)^{-1/2}U \Sigma V^T \\

&= (U \Sigma^2 U^T)^{-1/2}U \Sigma V^T \\

&= U \Sigma^{-1} U^T U \Sigma V^T \\

&= UV^T = O

\end{aligned}
$$

So, this means that $P_t^{(l)-1} = (M_t^{(l)}M_t^{(l)T})^{-1/2}$. To write it for the vectorized format, we can simply do:

$$
\begin{aligned}

w_t^{(l)} &= \text{vec}(W_t^{(l)}) \\

&= \text{vec}(W_{t-1}^{(l)} - \eta P_t^{(l)-1}M_t^{(l)}) \\

&= w_{t-1}^{(l)} - \eta \text{vec}(P_t^{(l)-1}M_t^{(l)}) \\

&= w_{t-1}^{(l)} - \eta (I \otimes P_t^{(l)-1})m_t^{(l)}

\end{aligned}
$$

using the linearity of vectorization and the identity $\text{vec}(AB) = (I \otimes A)\text{vec}(B)$. Therefore, we get that $D_t^{(l)-1} = (I \otimes P_t^{(l)-1})$.

The last step is to simply concatenate it for all layers. To do this, we can simply create a block-diagonal matrix $D_t^{-1}$ which contains all the per-layer preconditioners on the block-diagonals, i.e:

$$
D_t^{-1} = \left[\begin{matrix}  

D_t^{(1)-1} & 0 & \dots & 0 \\

0 & D_t^{(2)-1} & \dots & 0 \\

\vdots & \vdots & \ddots & \vdots \\

0 & 0 & \dots & D_t^{(L)-1}

\end{matrix}\right]
$$

This can be abbreviated using a direct sum: $D_t = \bigoplus_{l=1}^L D_t^{(l)-1}$. The complete update rule for all layers combined then becomes:

$$
w_t = w_{t-1} - \eta D^{-1}m_t
$$

Why is this relevant? This shows that the Muon update can be rewritten using a left preconditioner. In [this paper](https://arxiv.org/abs/2207.14484), the authors show that any left-preconditioned optimizer (with or without momentum) operates at the edge of stability. The difference is simply that the edge of stability is now in the effective Hessian.

## Hessian and Effective Hessian

The Hessian of a gradient based algorithm at time $t$ is simply

$$
H_t = \nabla^2_\theta\mathcal{L}(\theta_{t-1})
$$

Given a preconditioned gradient based algorithm (like Adam, RMSprop or Muon), there is also the notion of the effective Hessian:

$$
H_{t,eff} = P_t^{-1}H_t
$$

where $P_t^{-1}$ is the preconditioner at time $t$. It is also important to note the eigenvalues of $P_t^{-1}H_t$ are the same as the eigenvalues of $P_t^{-1/2}H_tP_t^{-1/2}$, since these are similar matrices.

The authors in [this paper](https://arxiv.org/abs/2103.00065) show that the edge of stability of vanilla gradient descent is when the sharpness of $H_t$ reaches $2/\eta$. They show that the same holds for the effective Hessian of RMSprop. The authors of [this paper](https://arxiv.org/abs/2207.14484) simply extend this notion and show that the same holds for any effective Hessian of a preconditioned algorithm. Additionally, they also compute thresholds for momentum, i.e. the only thing that changes is that the threshold then becomes $(2 + 2\beta)/\eta$ (for Heavy-Ball momentum). So, all that really remains to show is that this behavior also occurs with Muon.

Keep in mind that the phenomenon obseved in those papers relies on a third order Taylor expansion of the Hessian, so it is an approximation. Therefore, it often does not occur, for example when using Cross-Entroy loss instead of MSE. It could therefore be the case that it does not occur for Muon for some reason, so it is useful to find out.

## How all of this is combined in the code

The last thing that remains is to actually code all of this. I'll break this part up into multiple steps. For simplicity, I will only consider full-batch gradient descent, as the theory makes this assumption. However, similar behavior is also observed with batched GD.

So, given a dataset $\mathcal{D} = \{(x_i, y_i)\}_{i \le N}$, in each epoch we do:

### 1. Perform a Muon update

We first simply do the standard Muon update on the full dataset. This is simply done using PyTorch as is done usually.

### 2. Compute the sharpness

This part is a little more elaborate.

Since computing $H_{t,eff}$ is extremely computationally expensive, and mostly infeasible, we utilize Hessian-vector products to only ever compute $H_{t,eff}v$ for some vector $v$. This can be done quite efficiently. Since $A_t = D_t^{-1/2}H_tD_t^{-1/2}$ has the same eigenvalues as $H_{t,eff}$, we opt to use this to compute the sharpness as it has the nice property of being a symmetric matrix. This allows us to use specific algorithms like Lancosz to approximate the sharpness. Another option (simpler to implement) is using power iteration. Both of these approaches only ever require us to compute $A_tv$ for any vector $v$, and never require us to actually construct $A_t$.

This means that we need to compute $u = D_t^{-1/2}H_tD_t^{-1/2}v$. This can be split into three steps:

1. $v_1 = D_t^{-1/2}v$
2. $v_2 = H_t v_1$
3. $u = D_t^{-1/2}v_2$

Computing $D_t^{-1/2}v$ is a bit more complex, so lets first focus on $H_tv$. The key here is using the so-called Pearmutter trick. This allows us to compute $H_tv$ without having to compute $H_t$ itself. The Pearmutter trick can be written as follows:

$$
H_tv = \frac{d}{d\epsilon}\nabla_w\mathcal{L}(w_{t-1} + \epsilon v)\bigg\rvert_{\epsilon=0}
$$

More importantly, from an implementation point of view, this can simply be computed by differentiating the dot product of the gradient with the vector:

$$
H_t v = \nabla^2_w\mathcal{L}(w_{t-1})v = \nabla_w(\nabla_w\mathcal{L}(w_{t-1})^Tv)
$$

which can be done quite efficiently and quite easily with PyTorch.

Next, we have to be able to compute $D_t^{-1/2}$. Note that taking $D_t^p$ ($D$ is now the preconditioner) to any power $p$ can be done by simply taking the powers of all its block diagonal matrices, i.e. $D_t^p = \bigoplus_{l=1}^{L} D_t^{(l)p}$. This can be explained using spectral calculus, just take it as a given. This means we do not have to explicitly compute the entire $D_t$, but only have to compute the block diagonal matrices $D_t^{(l)}$. However, if we do not vectorize the parameters, we can simply compute $P_t^{(l)} = (M_t^{(l)}M_t^{(l)T})^{1/2}$. Note that this is equal to $(U \Sigma^2 U^T)^{1/2} = U \Sigma U^T$ where $M_t^{(l)} = U \Sigma V^T$ is the SVD. This is actually the exact diagonalization of $P_t^{(l)}$, so computing the power is simply $P_t^{(l)p} = U \Sigma^p U^T$ (keep in mind that this can be done since $D_t^{(l)} = (I \otimes P_t^{(l)})$ is also block-diagonal, specifically it can be rewritten as $\bigoplus P_t^{(l)}$). So, all we really have to do is set $p = -1/2$ and compute the SVD. Computing the SVD is computationally expensive, but since it is done per-layer we can get away with it if the width of the network is not extremely large.

## Note on using multiple optimizers

Since often not all parameters are rectangular, Muon is usually combined with other optimizers, like Adam or AdamW. This changes the preconditioner $D_t$, however it is still block-diagonal. We can simply add the preconditioner of Adam as a new block-diagonal matrix in $D_t$ and again compute it independently for its respective parameters.

## Muon's approximation

Ideal preconditioner (per-layer): $P = (MM^T)^{-1/2}$

Then, we get $O = PM = (MM^T)^{-1/2}M = (U\Sigma V^T V \Sigma U^T)^{-1/2}U\Sigma V^T = (U\Sigma^2 U^T)^{-1/2}U\Sigma V^T = U \Sigma^{-1} U^TU\Sigma V^T = UV^T$

But, the true Muon constructs an approximation of $O$, specifically $O \approx \hat{O} = U D V^T$, where $D$ is a diagonal matrix with diagonal entries close to 1 (e.g. $[0.3, 1.7]$ or so). So, what if we replace $\Sigma$ with a diagonal matrix $S$ such that $S^{-1}\Sigma = D$. I.e., since $W_{t+1} = W_t - \eta \hat{O}_t$, we gotta have $US^{-1}\Sigma V^T = \hat{O_t} = (W_t - W_{t+1})/\eta$. Then the exact preconditioner (of the approximation) becomes $U S^{-1} U^T$, since $(U S^{-1} U^T)M = U S^{-1} U^T U\Sigma V^T = US^{-1}\Sigma V^T = U D V^T = \hat{O}$.

So how do we find $S$? Suppose we have the true update matrix $\hat{O}$. We can simply find $U$ and $\Sigma$ via SVD of $M = U\Sigma V^T$. We then need to find $D$. Since we have $U$ and $V$, $D$ can be found by doing $D = U^T\hat{O} V$. What remains is to simply solve $S^{-1}\Sigma = D$. All of these are diagonal matrices, so each of these amounts to a very simple linear equation. I.e. $S^{-1} = D\Sigma^{-1}$, so $S = \Sigma D^{-1}$

## 
$$
\rho(M) = U(aS + bS^3 + cS^5)V^T = UDV^T
$$


$$
\begin{aligned}

&\rho(\rho(M)) = a \rho(M) + b(\rho(M)\rho(M)^T)\rho(M) + c(\rho(M)\rho(M)^T)^2\rho(M)\\

&= a UDV^T + b(UDV^TVDU^T)UDV^T + c(UDV^TVDU^T)^2UDV^T\\

&= aUDV^T + b(UD^2U^T)UDV^T + c(UD^2U^T)^2UDV^T\\

&= aUDV^T + bUD^3V^T + cUD^5V^T\\

&= U(aD + bD^3 + cD^5)V^T \\

&= U \tilde{D} V^T

\end{aligned}
$$

$$
\begin{aligned}

D^pv &= \left( \bigoplus_{i=1}^L (I \otimes P_i) \right) v \\

&= \bigoplus_{i=1}^L(I \otimes P_i)^pv_i \\

&= \bigoplus_{i=1}^L(I \otimes P^p_i)v_i \\

&= \bigoplus_{i=1}^L\text{vec}(P^p_iV_i) \\


\end{aligned}
$$

$$
\begin{aligned}



\end{aligned}
$$