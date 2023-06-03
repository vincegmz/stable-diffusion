import torch
import numpy as np 
from .random_util import get_generator

class ConsistentSolverSampler(object):
    def __init__(self, model,   training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        diffusion_rho=7,
        sigma_max=0.002,
        sigma_min=80,
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=42,
        ts="", **kwargs):
        super().__init__()
        self.model = model
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(model.device)
        self.register_buffer('alphas_cumprod', to_torch(model.alphas_cumprod))
        self.generator = generator
        self.clip_denoised = clip_denoised
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.sampler = sampler
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax    
        self.s_noise = s_noise
        self.steps = steps
        self.model_path = model_path
        self.seed = seed
        self.ts = ts
        self.training_mode = training_mode
        self.diffusion_rho = diffusion_rho
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min

    

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               x_T=None,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwself
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)

        # print(f'Data shape for DPM-Solver sampling is {size}, sampling steps {S}')

        device = self.model.betas.device
        x = karras_sample(self.model,
            (self.batch_size, 3, self.image_size, self.image_size),
            steps=self.steps,
            device=device,
            clip_denoised=self.clip_denoised,
            sampler=self.sampler,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            s_churn=self.s_churn,
            s_tmin=self.s_tmin,
            s_tmax=self.s_tmax,
            s_noise=self.s_noise,
            generator=self.generator,
            ts=self.ts,
            conditions = conditioning)
        
        return x.to(device), None

def karras_sample(
    model,
    shape,
    steps=40,
    clip_denoised=True,
    progress=False,
    callback=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80,  # higher for highres?
    rho=7.0,
    sampler="heun",
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    diffusion_rho = 7,
    generator=None,
    x_T = None,
    ts=None,
    conditions = None,
):

    if generator is None:
        generator = get_generator("dummy")

    if sampler == "progdist":
        sigmas = get_sigmas_karras(steps + 1, sigma_min, sigma_max, rho, device=device)
    else:
        sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)

    x_T = generator.randn(*shape, device=device) * sigma_max

    sample_fn = stochastic_iterative_sampler

    if sampler in ["heun", "dpm"]:
        sampler_self = dict(
            s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise
        )
    elif sampler == "multistep":
        sampler_self = dict(
            ts=ts, t_min=sigma_min, t_max=sigma_max, rho=diffusion_rho, steps=steps
        )
    else:
        sampler_self = {}

    def denoiser(x_t, sigma):
        _, denoised = model.apply_model(x_t, sigma, conditions)
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised

    x_0 = sample_fn(
        denoiser,
        x_T,
        sigmas,
        generator,
        progress=progress,
        **sampler_self,
    )
    return x_0.clamp(-1, 1)

@torch.no_grad()
def stochastic_iterative_sampler(
    distiller,
    x,
    sigmas,
    generator,
    ts,
    progress=False,
    callback=None,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros([1])]).to(device)
