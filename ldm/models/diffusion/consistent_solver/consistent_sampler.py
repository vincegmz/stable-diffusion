import torch
import numpy as np 
from .random_util import get_generator
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor
from ldm.util import default
import torch.nn
class ConsistentSolverSampler(object):
    def __init__(self, model, training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        diffusion_rho=7,
        sigma_max=80,
        sigma_min=0.002,
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=42,
        ts="0,22,39", **kwargs):
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
        self.ts = tuple(int(x) for x in ts.split(","))
        self.training_mode = training_mode
        self.diffusion_rho = diffusion_rho
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self):
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.model.num_timesteps #'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))


    @torch.no_grad()
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
    
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
        self.make_schedule()
        # print(f'Data shape for DPM-Solver sampling is {size}, sampling steps {S}')

        device = self.model.betas.device
        x = self.karras_sample(
            size,
            device,
            conditions = conditioning)
        
        return x.to(device), None
    def karras_sample(self,
        shape,
        device,
        conditions = None,
    ):

        # if generator is None:
        #     generator = get_generator("dummy")

        x_T = torch.randn(*shape, device=device)
        # x_T should be multiplied by sigma_max?
        # x_T = generator.randn(*shape,device = device)*sigma_max

        sample_fn = self.stochastic_iterative_sampler

        if self.sampler == "multistep":
            sampler_self = dict(
                ts=self.ts, t_min=self.sigma_min, t_max=self.sigma_max, rho=self.diffusion_rho, steps=self.steps
            )
        else:
            raise NotImplementedError('only multistep consistent sampling supported')

        # def denoiser(x_t, sigma):
        #     noise = self.model.apply_model(x_t,sigma,conditions)

        #     if self.clip_denoised:
        #         distiller_target.clamp_(-1., 1.)
        #     _, denoised = self.model.apply_model(x_t, sigma, conditions)
        #     if self.clip_denoised:
        #         denoised = denoised.clamp(-1, 1)
        #     return denoised
        
        x_0 = sample_fn(
            x_T,
            conditions,
            **sampler_self,
        )
        return x_0.clamp(-1, 1)

    @torch.no_grad()
    def stochastic_iterative_sampler(self,
        x,
        conditions,
        ts = None,
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
            t = round(t)- 1
            out = self.model.apply_model(x, t * s_in,conditions,'online')
            if self.model.parameterization == "eps":
                x0= self.model.predict_start_from_noise(x, t=(t*s_in).to(torch.int64), noise=out)
            elif self.parameterization == "x0":
                x0 = out
            else:
                raise NotImplementedError()
            next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
            next_t = round(next_t) - 1
            next_t = np.clip(next_t, t_min, t_max)
            x = self.model.q_sample(x0,(next_t*s_in).to(torch.int64))

        return x


    
