import torch
import torch.nn as nn
import numpy as np
import math
import jax.numpy as jnp
import abc


def ce_loss(outputs, targets, cl_g):
    if cl_g.n_class == 1:
        loss = nn.BCEWithLogitsLoss()
    else:
        loss = nn.CrossEntropyLoss()
    return loss(outputs, targets)

# Taken and updated from https://github.com/yang-song/score_sde_pytorch/blob/main/sde_lib.py
# def sde(x, t, beta_0, beta_1):
#     beta_t = beta_0 + t * (beta_1 - beta_0)
#     drift = -0.5 * beta_t[:, None, None, None] * x
#     diffusion = torch.sqrt(beta_t)
#     return drift, diffusion

# def rev_sde(x, t, score_fn, beta_0, beta_1, probability_flow, cl_g=None, cl_s=None, target=None):
#     """Create the drift and diffusion functions for the reverse SDE/ODE."""
#     drift, diffusion = sde(x, t, beta_0, beta_1)
#     score = score_fn(x, t)

#     # if classifier guidance
#     if cl_g is not None:
#         with torch.enable_grad():
#             x.requires_grad = True
#             cl_out = cl_g(x.view(x.shape[0],-1), t)
#             cl_loss = ce_loss(cl_out, target)
#             grad = torch.autograd.grad(cl_loss, x)[0]
#             if cl_s is not None:
#                 score += cl_s * grad
#             else:
#                 score += grad
    
#     drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if probability_flow else 1.)
#     # Set the diffusion function to zero for ODEs.
#     diffusion = 0. if probability_flow else diffusion
#     return drift, diffusion

def em_predictor(x, t, score_fn, sde, probability_flow=False, cl_g=None, cl_s=None, target=None, given=None, all_mods=None):
    dt = -1. / sde.N
    z = torch.randn_like(x)
    rev_sde = sde.reverse(score_fn, probability_flow)
    drift, diffusion = rev_sde.sde(x, t, cl_g, cl_s, target, given=given, all_mods=all_mods)
    x_mean = x + drift * dt
    x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
    return x, x_mean

def corrector(x, t, score_fn, sde, n_steps, target_snr, cl_g=None, cl_s=None, target=None, given=None, all_mods=None):
    # cl_g = None
    if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
        timestep = (t * (sde.N - 1) / sde.T).long()
        alpha = sde.alphas.to(t.device)[timestep]
    else:
       alpha = torch.ones_like(t)

    for i in range(n_steps):
        grad = score_fn(x, t)

        if cl_g is not None and given is not None and given:
            with torch.enable_grad():
                predicted_mods = ''.join([m for m in all_mods if m not in given])

                if ('0' in given and '1' in predicted_mods) or ('1' in given and '0' in predicted_mods):
                    mod1, mod2 = '0', '1'
                    new_x = torch.cat([x[:,int(mod1)-int(all_mods[0]),:,:].unsqueeze(1), x[:,int(mod2)-int(all_mods[0]),:,:].unsqueeze(1)], dim=1)
                    new_x.requires_grad = True
                    cl_out = cl_g['01'](new_x.view(new_x.shape[0],-1), t)
                    cls_grad = torch.autograd.grad(cl_out.mean(), new_x)[0]
                    grad[:,int(mod1)-int(all_mods[0])] -= cl_s * cls_grad[:,0]
                    grad[:,int(mod2)-int(all_mods[0])] -= cl_s * cls_grad[:,1]
				
                if ('0' in given and '2' in predicted_mods) or ('2' in given and '0' in predicted_mods):
                    mod1, mod2 = '0', '2'
                    new_x = torch.cat([x[:,int(mod1)-int(all_mods[0]),:,:].unsqueeze(1), x[:,int(mod2)-int(all_mods[0]),:,:].unsqueeze(1)], dim=1)
                    new_x.requires_grad = True
                    cl_out = cl_g['02'](new_x.view(new_x.shape[0],-1), t)
                    cls_grad = torch.autograd.grad(cl_out.mean(), new_x)[0]
                    grad[:,int(mod1)-int(all_mods[0])] -= cl_s * cls_grad[:,0]
                    grad[:,int(mod2)-int(all_mods[0])] -= cl_s * cls_grad[:,1]

                if ('1' in given and '2' in predicted_mods) or ('2' in given and '1' in predicted_mods):
                    mod1, mod2 = '1', '2'
                    new_x = torch.cat([x[:,int(mod1)-int(all_mods[0]),:,:].unsqueeze(1), x[:,int(mod2)-int(all_mods[0]),:,:].unsqueeze(1)], dim=1)
                    new_x.requires_grad = True
                    cl_out = cl_g['12'](new_x.view(new_x.shape[0],-1), t)
                    cls_grad = torch.autograd.grad(cl_out.mean(), new_x)[0]
                    grad[:,int(mod1)-int(all_mods[0])] -= cl_s * cls_grad[:,0]
                    grad[:,int(mod2)-int(all_mods[0])] -= cl_s * cls_grad[:,1]

        noise = torch.randn_like(x)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        x_mean = x + step_size[:, None, None, None] * grad
        x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        # x_mean = x + 0.01 * grad
        # x = x_mean + 0.5 * np.sqrt(0.01 * 2) * noise

    return x, x_mean

# def marginal_prob(x, t, beta_0, beta_1):
#     log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
#     # print('marginal prob x.shape: ',x.shape, 't: ', t.shape, 'logmean: ', log_mean_coeff[:, None, None, None].shape, flush=True)
#     mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
#     std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
#     return mean, std

def uncond_sampler(sample_shape, model, device, sde, eps=1e-3, probability_flow=False, pc=False, n_steps=1, target_snr=0.16, cl_g=None, cl_s=None, target=None):
    with torch.no_grad():
        # Initial sample
        x = sde.prior_sampling(sample_shape).to(device)
        timesteps = torch.linspace(sde.T, eps,sde.N, device=device)

        for i in range(sde.N):
            t = timesteps[i]
            vec_t = torch.ones(sample_shape[0], device=t.device) * t
            if pc:
                x, x_mean = corrector(x, vec_t, model, sde, n_steps, target_snr, cl_g, cl_s, target)
            x, x_mean = em_predictor(x, vec_t, model, sde, probability_flow, cl_g, cl_s, target)

        return x_mean


def likelihood_importance_cum_weight(t, beta_0, beta_1, eps=1e-5):
    exponent1 = 0.5 * eps * (eps - 2) * beta_0 - 0.5 * eps ** 2 * beta_1
    exponent2 = 0.5 * t * (t - 2) * beta_0 - 0.5 * t ** 2 * beta_1
    term1 = jnp.where(jnp.abs(exponent1) <= 1e-3, -exponent1, 1. - jnp.exp(exponent1))
    term2 = jnp.where(jnp.abs(exponent2) <= 1e-3, -exponent2, 1. - jnp.exp(exponent2))
    return 0.5 * (-2 * jnp.log(term1) + 2 * jnp.log(term2) + beta_0 * (-2 * eps + eps ** 2 - (t - 2) * t) + beta_1 * (-eps ** 2 + t ** 2))

def sample_importance_weighted_time_for_likelihood(shape, beta_0, beta_1, quantile=None, eps=1e-5, steps=100, T=1):
    Z = likelihood_importance_cum_weight(T, beta_0, beta_1, eps)
    if quantile is None:
      quantile = torch.distributions.uniform.Uniform(0,Z.item()).sample((shape,)).numpy()
    lb = jnp.ones_like(quantile) * eps
    ub = jnp.ones_like(quantile) * T

    for i in range(steps):
        mid = (lb + ub) / 2.
        value = likelihood_importance_cum_weight(mid, beta_0, beta_1, eps=eps)
        lb = jnp.where(value <= quantile, mid, lb)
        ub = jnp.where(value <= quantile, ub, mid)
    return (lb + ub) / 2.

def loss_fn(batch, score_fn, sde, reduce_mean=True, likelihood_weighting=True, eps=1e-5, im_sample=False):
    """Compute the loss function.

    Args:
      model: A score model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    if likelihood_weighting and im_sample:
        t = torch.tensor(np.array(sample_importance_weighted_time_for_likelihood(batch.shape[0], sde.beta_0, sde.beta_1, T=sde.T))).to(batch.device)
    else:
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_data, t)

    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
        if im_sample:
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss = torch.mean(losses)
    return loss


# Taken and adopted from https://github.com/yang-song/score_sde_pytorch/blob/main/sde_lib.py
"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, N):
    """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.N = N

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  @abc.abstractmethod
  def prior_sampling(self, shape):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

  @abc.abstractmethod
  def prior_logp(self, z):
    """Compute log-density of the prior distribution.

    Useful for computing the log-likelihood via probability flow ODE.

    Args:
      z: latent code
    Returns:
      log probability density
    """
    pass

  def discretize(self, x, t):
    """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.

    Args:
      x: a torch tensor
      t: a torch float representing the time step (from 0 to `self.T`)

    Returns:
      f, G
    """
    dt = 1 / self.N
    drift, diffusion = self.sde(x, t)
    f = drift * dt
    G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
    return f, G

  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
        def __init__(self):
            self.N = N
            self.probability_flow = probability_flow

        @property
        def T(self):
            return T

        def sde(self, x, t, cl_g=None, cl_s=None, target=None, given=None, all_mods=None):
            """Create the drift and diffusion functions for the reverse SDE/ODE."""
            drift, diffusion = sde_fn(x, t)
            score = score_fn(x, t)

            # if classifier guidance
            if cl_g is not None and given is not None and given:
                with torch.enable_grad():
                    predicted_mods = ''.join([m for m in all_mods if m not in given])

                    if ('0' in given and '1' in predicted_mods) or ('1' in given and '0' in predicted_mods):
                        mod1, mod2 = '0', '1'
                        new_x = torch.cat([x[:,int(mod1)-int(all_mods[0]),:,:].unsqueeze(1), x[:,int(mod2)-int(all_mods[0]),:,:].unsqueeze(1)], dim=1)
                        new_x.requires_grad = True
                        cl_out = cl_g['01'](new_x.view(new_x.shape[0],-1), t)
                        cls_grad = torch.autograd.grad(cl_out.mean(), new_x)[0]
                        score[:,int(mod1)-int(all_mods[0])] -= cl_s * cls_grad[:,0]
                        score[:,int(mod2)-int(all_mods[0])] -= cl_s * cls_grad[:,1]
                    
                    if ('0' in given and '2' in predicted_mods) or ('2' in given and '0' in predicted_mods):
                        mod1, mod2 = '0', '2'
                        new_x = torch.cat([x[:,int(mod1)-int(all_mods[0]),:,:].unsqueeze(1), x[:,int(mod2)-int(all_mods[0]),:,:].unsqueeze(1)], dim=1)
                        new_x.requires_grad = True
                        cl_out = cl_g['02'](new_x.view(new_x.shape[0],-1), t)
                        cls_grad = torch.autograd.grad(cl_out.mean(), new_x)[0]
                        score[:,int(mod1)-int(all_mods[0])] -= cl_s * cls_grad[:,0]
                        score[:,int(mod2)-int(all_mods[0])] -= cl_s * cls_grad[:,1]

                    if ('1' in given and '2' in predicted_mods) or ('2' in given and '1' in predicted_mods):
                        mod1, mod2 = '1', '2'
                        new_x = torch.cat([x[:,int(mod1)-int(all_mods[0]),:,:].unsqueeze(1), x[:,int(mod2)-int(all_mods[0]),:,:].unsqueeze(1)], dim=1)
                        new_x.requires_grad = True
                        cl_out = cl_g['12'](new_x.view(new_x.shape[0],-1), t)
                        cls_grad = torch.autograd.grad(cl_out.mean(), new_x)[0]
                        score[:,int(mod1)-int(all_mods[0])] -= cl_s * cls_grad[:,0]
                        score[:,int(mod2)-int(all_mods[0])] -= cl_s * cls_grad[:,1]
                    
            drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
            # Set the diffusion function to zero for ODEs.
            diffusion = 0. if self.probability_flow else diffusion
            return drift, diffusion

        def discretize(self, x, t):
            """Create discretized iteration rules for the reverse diffusion sampler."""
            f, G = discretize_fn(x, t)
            rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
            rev_G = torch.zeros_like(G) if self.probability_flow else G
            return rev_f, rev_G

    return RSDE()


class VPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
    return logps

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    f = torch.sqrt(alpha)[:, None, None, None] * x - x
    G = sqrt_beta
    return f, G


class subVPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct the sub-VP SDE that excels at likelihoods.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
    diffusion = torch.sqrt(beta_t * discount)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
    std = 1 - torch.exp(2. * log_mean_coeff)
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.


class VESDE(SDE):
  def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
    """Construct a Variance Exploding SDE.

    Args:
      sigma_min: smallest sigma.
      sigma_max: largest sigma.
      N: number of discretization steps
    """
    super().__init__(N)
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.beta_0 = self.sigma_min
    self.beta_1 = self.sigma_max
    self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    drift = torch.zeros_like(x)
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                device=t.device))
    return drift, diffusion

  def marginal_prob(self, x, t):
    std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    mean = x
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape) * self.sigma_max

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)

  def discretize(self, x, t):
    """SMLD(NCSN) discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    sigma = self.discrete_sigmas.to(t.device)[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                 self.discrete_sigmas[timestep - 1].to(t.device))
    f = torch.zeros_like(x)
    G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
    return f, G
