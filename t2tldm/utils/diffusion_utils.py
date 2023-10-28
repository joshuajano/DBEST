import os 
from tqdm import tqdm
import numpy as np
import torchvision.utils as tvu

from typing import Optional, Union, Tuple, List, Callable, Dict

import torch
def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
@torch.no_grad()
def run_ddpm_q_sample(z0, t, noise, SQRT_ALPHAS_CUMPROD, SQRT_ONE_MINUS_ALPHAS_CUMPROD):
    return (extract_into_tensor(SQRT_ALPHAS_CUMPROD, t, z0.shape) * z0 +
            extract_into_tensor(SQRT_ONE_MINUS_ALPHAS_CUMPROD, t, z0.shape) * noise)

def run_ddpm_clean_q_sample(z0_noisy, t, noise, SQRT_ALPHAS_CUMPROD, SQRT_ONE_MINUS_ALPHAS_CUMPROD):
    clean_z0 = (z0_noisy - (extract_into_tensor(SQRT_ONE_MINUS_ALPHAS_CUMPROD, t, z0_noisy.shape) * noise)) /\
                extract_into_tensor(SQRT_ALPHAS_CUMPROD, t, z0_noisy.shape)
    return clean_z0

def run_ddpm_q_sample_grad(z0, t, noise, SQRT_ALPHAS_CUMPROD, SQRT_ONE_MINUS_ALPHAS_CUMPROD):
    return (extract_into_tensor(SQRT_ALPHAS_CUMPROD.detach(), t, z0.shape) * z0 +
            extract_into_tensor(SQRT_ONE_MINUS_ALPHAS_CUMPROD.detach(), t, z0.shape) * noise)
@torch.no_grad()
def next_step(scheduler, model_output: Union[torch.FloatTensor, np.ndarray], 
            timestep: int, sample: Union[torch.FloatTensor, np.ndarray], 
            num_inference_steps=50):
        if scheduler.num_inference_steps == None:
            scheduler.num_inference_steps = num_inference_steps
        timestep, next_timestep = min(timestep - scheduler.config.num_train_timesteps // 
                                scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = scheduler.alphas_cumprod[timestep] if timestep >= 0 else scheduler.final_alpha_cumprod
        alpha_prod_t_next = scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
@torch.no_grad()
def get_noise_pred_single(model, zt, t, context):
        noise_pred = model.unet(zt, t, encoder_hidden_states=context)["sample"]
        return noise_pred

@torch.no_grad()
def run_ddim_q_sample(model, scheduler, z0, context, 
                    NUM_DDIM_STEPS = 50,
                    COND_Q_SAMPLE = True,
                    save_dir= 'debug/original_null_inversion/', 
                    num_inference_steps=50):
    save_path = os.path.join(save_dir, 'inversion')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_zt = [z0]
    zt = z0.clone().detach()
    with tqdm(total=NUM_DDIM_STEPS, desc=f"DDIM inversion process ") as progress_bar:
        for i in range(NUM_DDIM_STEPS):
            t = model.scheduler.timesteps[len(model.scheduler.timesteps) - i - 1]
            if COND_Q_SAMPLE:
                et = get_noise_pred_single(model, zt, t, cond_embeddings)
            else:
                et = get_noise_pred_single(model, zt, t, uncond_embeddings)
        
            # et_uncond = get_noise_pred_single(model, zt, t, uncond_embeddings)
            # et_cond = get_noise_pred_single(model, zt, t, cond_embeddings)
            # et = et_uncond + 1 * (et_cond - et_uncond)
            #--Get the next zt
            zt = next_step(scheduler, et, t, zt)
            tvu.save_image((zt + 1) * 0.5, os.path.join(save_path, f'z{i}.png'))
            all_zt.append(zt)
            progress_bar.update(1)
    return all_zt, zt