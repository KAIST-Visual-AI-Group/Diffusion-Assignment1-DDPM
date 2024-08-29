from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class VarianceScheduler(nn.Module):
    def __init__(self, num_steps, beta_1=1e-4, beta_T=0.02, mode="linear"):
        super().__init__()
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == "quad":
            betas = torch.linspace(beta_1**0.5, beta_T**0.5, num_steps) ** 2
        elif mode == "cosine":
            cosine_s = 8e-3
            timesteps = torch.arange(num_steps + 1) / num_steps + cosine_s
            alphas = timesteps / (1 + cosine_s) * np.pi / 2
            alphas = torch.cos(alphas).pow(2)
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def uniform_sample_t(self, batch_size) -> torch.LongTensor:
        ts = np.random.choice(np.arange(self.num_steps), batch_size)
        return torch.from_numpy(ts)
    

class DiffusionModule(nn.Module):
    def __init__(self, network, var_scheduler, **kwargs):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler

    def get_loss(self, x0, class_label=None, noise=None):
        ######## TODO ########
        # DO NOT change the code outside this part.
        # compute noise matching loss.
        B = x0.shape[0]
        timestep = self.var_scheduler.uniform_sample_t(B, self.device)        
        loss = x0.mean()
        ######################
        return loss
    
    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def image_resolution(self):
        return self.network.image_resolution

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        return_traj=False,
        class_label: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 1.0,
    ):
        x_T = torch.randn([batch_size, 3, self.image_resolution, self.image_resolution]).to(self.device)

        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:

            ######## TODO ########
            # Assignment 2. Implement the classifier-free guidance.
            # Specifically, given a tensor of shape (batch_size,) containing class labels,
            # create a tensor of shape (2*batch_size,) where the first half is filled with zeros (i.e., null condition).
            assert class_label is not None
            assert len(class_label) == batch_size, f"len(class_label) != batch_size. {len(class_label)} != {batch_size}"
            raise NotImplementedError("TODO")
            #######################

        traj = [x_T]
        for t in tqdm(self.var_scheduler.timesteps):
            x_t = traj[-1]
            if do_classifier_free_guidance:
                ######## TODO ########
                # Assignment 2. Implement the classifier-free guidance.
                raise NotImplementedError("TODO")
                #######################
            else:
                noise_pred = self.network(x_t, timestep=t.to(self.device))

            x_t_prev = self.var_scheduler.step(x_t, t, noise_pred)

            traj[-1] = traj[-1].cpu()
            traj.append(x_t_prev.detach())

        if return_traj:
            return traj
        else:
            return traj[-1]
