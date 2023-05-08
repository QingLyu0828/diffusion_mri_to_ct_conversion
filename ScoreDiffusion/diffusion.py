import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from scipy import integrate

device = torch.device('cuda:0')

def marginal_prob_std(t, sigma):
    """ Compute standard deviation of conditional Gaussian distribution at time t.  
    SDE: dx = sigma^t * dw  t belongs to [0,1]
    p_{0t}(x(t) | x(0)) = N(m,var) = N(x(t); x(0), 1 / (2log(sigma)) * (sigma^(2t)-1) * I)
    
    """
    t = torch.tensor(t, device=device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
    """ Compute diffusion coefficient at time t """
    return torch.tensor(sigma**t, device=device)

def loss_fn(score_model, condition, x, marginal_prob_std, eps=1e-5):
    """ The loss function for training score-based generative models
    
    Args:
        score_model: A PyTorch model instance that represents a time-dependent
            score-based model
        x: A mini-batch of training data
        condition: input image as condition
        marginal_prob_std: A function that gives the standard deviation of
            the perturbation kernel
        eps: A tolerance value for numerical stability
    """
    # Step 1: randomly generate time t from [0.0001, 0.9999]
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    
    # Step 2: sampling a perturbed_x sample from data distribtion p_t(x) based on the reparameterization trick
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]
    
    # Step 3: adding noised sample into the score network to estimate score
    score = score_model(torch.cat([perturbed_x, condition], dim=1), random_t)
    
    # Step 4: Computing score matching loss
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
    
    return loss
    

class EMA(nn.Module):
    def __init__(self, model, decay=0.9999):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        # self.module = deepcopy(model)       
        self.module.eval()
        self.decay = decay
        
    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))
                
    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)
        
    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
    
    
def euler_sampler(score_model, condition, marginal_prob_std, diffusion_coeff, batch_size=64, num_steps=1000, eps=1e-3):
    """
       SDE: dx = f(x,t)dt + g(t)dw
       Reverse-time SDE: dx = [f(x,t) - g(t)**2 * score]dt + g(t)dw_bar
       In this case, omit f(x,t) and choose SDE: dx = sigma*t * dw  t belongs to [0,1]
       Reverse-time SDE: dx = -sigma**{2t} * score * dt + sigma**t * dw_bar
       
       To sample from time-dependent score-based model, first draw a sample from the prior
       distribution p_1 ~ N(x; 0, 0.5*(sigma**2 - 1)*I), then solve the reverse-time SDE
       via Euler-Maruyama approach. Replacing dw with z ~ N(0, g(t)**2 * dt * I),
       we can obtain the iteration rule: 
           x_{t-dt} = x_t + sigma**{2t} * score * dt + sigma**t * sqrt(dt) * z_t, where
           z_t ~ N(0,I)
    """
    
    # Step 1: define start time t=1 and random samples from prior data distribution
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, 512, 512, device=device) * marginal_prob_std(t)[:, None, None, None]
    
    # Step 2: define reverse time grid and time invervals
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    
    # Step 3: solve reverse time SDE via Euler-Maruyama approach
    x = init_x
    with torch.no_grad():
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * score_model(torch.cat([x, condition], dim=1), batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
          
    # Step 4: select final step expectation as a sampler
    return mean_x
    
    
def pc_sampler(score_model, condition, marginal_prob_std, diffusion_coeff, batch_size=64, snr=0.16, num_steps=1000, eps=1e-3):
    """ Generate samplers from score-based models with Predictor-Corrector method.
    
    Parameters
    ----------
    score_model : A PyTorch model instance that represents a time-dependent
            score-based model.
    marginal_prob_std : A function that gives the standard deviation of
            the perturbation kernel
    diffusion_coeff : A function that gives the diffusion coefficient
    batch_size : default: 64
    snr : signal-to-noise-ratio, default: 0.16
    """
    # Step 1: define start time t=1 and random samples from prior data distribution
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, 512, 512, device=device) * marginal_prob_std(t)[:, None, None, None]
        
    # Step 2: define reverse time grid and time invervals
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    
    # Step 3: alternatively use Langevin sampling and reverse-time SDE with Euler approach to solve
    x = init_x
    with torch.no_grad():
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            
            # Corrector step (Langevin MCMC)
            grad = score_model(torch.cat([x, condition], dim=1), batch_time_step)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            # print(f"{langevin_step_size=}")
        
            for _ in range(10):
                x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)
                grad = score_model(torch.cat([x, condition], dim=1), batch_time_step)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = np.sqrt(np.prod(x.shape[1:]))
                langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
                # print(f"{langevin_step_size=}")
            
            # Predictor step (Euler-Maruyama)
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * score_model(torch.cat([x, condition], dim=1), batch_time_step) * step_size
            x = mean_x + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)
   
        # Step 4: select final step expectation as a sampler
        return mean_x
    
def ode_sampler(score_model, condition, marginal_prob_std, diffusion_coeff, batch_size=64, atol=1e-5, rtol=1e-5, z=None, eps=1e-3):
    """ Generate samplers from score-based models with ODE method """
    
    # Step 1: define start time t=1 and initial x
    t = torch.ones(batch_size, device=device)
    if z is None:
        init_x = torch.randn(batch_size, 1, 512, 512, device=device) * marginal_prob_std(t)[:, None, None, None]
    else:
        init_x = z
    shape = init_x.shape
    
    # Step 2: define score estimation function and ODE function
    def score_eval_wrapper(sample, time_steps):
        """ A Wrapper of the score-based model for use by the ODE solver """
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))
        with torch.no_grad():
            score = score_model(torch.cat([sample, condition], dim=1), time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)
    
    def ode_func(t, x):
        """ The ODE function for use by the ODE solver """
        time_steps = np.ones((shape[0],)) * t
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        return -0.5 * (g**2) * score_eval_wrapper(x, time_steps)
    
    # Step 3: using ODE to solve value at t=eps
    res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')
    # print(f"Number of function evaluations: {res.nfev}")
    
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)
    
    return x
        
    
    
    