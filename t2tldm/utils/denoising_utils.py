import os
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
import torchvision.utils as tvu
# from baselines.prompt2prompt import ptp_utils
from typing import Optional, Union, Tuple, List, Callable, Dict
def prev_step(scheduler, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, 
                    sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        alpha_prod_t = scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
@torch.no_grad()
def run_ddim_mixture_p_sample_with_random_gs(model, scheduler, zT, context, 
                    NUM_DDIM_STEPS = 50,
                    num_inference_steps=50, 
                    start_time = 50, guidance_scale=7.5,
                    alpha_mix =0.2, 
                    save_dir= 'debug/original_null_inversion/', ):
    save_path = os.path.join(save_dir, 'generative_mixture')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    zt = zT.clone().detach()
    gs_src = torch.rand_like(zt)
    layer_gs_src = nn.Sigmoid()(gs_src)
    gs_tgt = torch.rand_like(zt)
    layer_gs_tgt = nn.Sigmoid()(gs_tgt)

    model.scheduler.set_timesteps(num_inference_steps)
    with tqdm(total=NUM_DDIM_STEPS, desc=f"DDIM generative process ") as progress_bar:
        for i, t in enumerate(model.scheduler.timesteps[-start_time:]):
            zt_cat = torch.cat([zt] * 3)
            noise_pred = model.unet(zt_cat, t, encoder_hidden_states=context.clone())["sample"]
            et_uncond, et_src_cond, et_tgt_cond = noise_pred.chunk(3)
            # et = et_cond 
            # et = et_uncond + \
            #     (1 - alpha_mix) * guidance_scale * (et_src_cond - et_uncond) +\
            #     (alpha_mix) * guidance_scale     * (et_tgt_cond - et_uncond)
            et = et_uncond + \
                ( layer_gs_src) * guidance_scale  * (et_src_cond - et_uncond) +\
                (1 - layer_gs_tgt) * guidance_scale   * (et_tgt_cond - et_uncond)
            zt = model.scheduler.step(et, t, zt)["prev_sample"]
            tvu.save_image((zt + 1) * 0.5, os.path.join(save_path, f'z{i}.png'))
            progress_bar.update(1)
    return zt
@torch.no_grad()
def run_custom_mixture_p_sample_with_norm_gs_p2p(model, scheduler, controller, zT, context,
                    NUM_DDIM_STEPS = 50,
                    num_inference_steps=50, 
                    start_time = 50, guidance_scale=7.5,
                    alpha_mix =0.2, 
                    save_dir= 'debug/original_null_inversion/', ):
    ptp_utils.register_attention_control(model, controller)
    save_path = os.path.join(save_dir, 'generative_mixture')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    zt = zT.clone().detach()
    scheduler.set_timesteps(num_inference_steps)

    with tqdm(total=NUM_DDIM_STEPS, desc=f"Custom generative process ") as progress_bar:
        for i, t in enumerate(scheduler.timesteps[-start_time:]):
            zt_cat = torch.cat([zt] * 3)
            noise_pred = model.unet(zt_cat, t, encoder_hidden_states=context.clone())["sample"]
            et_uncond, et_src_cond, et_tgt_cond = noise_pred.chunk(3)
            # et = et_cond 
            et = et_uncond + \
                (1 - alpha_mix) * guidance_scale *( (et_src_cond - et_uncond)/ torch.norm(et_src_cond - et_uncond) * torch.norm(et_uncond)) +\
                (alpha_mix) * guidance_scale  * ((et_tgt_cond - et_uncond)/ torch.norm(et_tgt_cond - et_uncond) * torch.norm(et_uncond))
            zt = scheduler.step(et, t, zt)["prev_sample"]
            tvu.save_image((zt + 1) * 0.5, os.path.join(save_path, f'z{i}.png'))
            progress_bar.update(1)
    return zt
@torch.no_grad()
def run_custom_mixture_p_sample_with_norm_gs(model, scheduler, zT, context,
                    NUM_DDIM_STEPS = 50,
                    num_inference_steps=50, 
                    start_time = 50, guidance_scale=7.5,
                    alpha_mix =0.2, 
                    save_dir= 'debug/original_null_inversion/', ):
    
    save_path = os.path.join(save_dir, 'generative_mixture')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    zt = zT.clone().detach()
    scheduler.set_timesteps(num_inference_steps)
    
    with tqdm(total=NUM_DDIM_STEPS, desc=f"Custom generative process ") as progress_bar:
        for i, t in enumerate(scheduler.timesteps[-start_time:]):
            zt_cat = torch.cat([zt] * 3)
            noise_pred = model.unet(zt_cat, t, encoder_hidden_states=context.clone())["sample"]
            et_uncond, et_src_cond, et_tgt_cond = noise_pred.chunk(3)
            # et = et_cond 
            et = et_uncond + \
                (1 - alpha_mix) * guidance_scale *( (et_src_cond - et_uncond)/ torch.norm(et_src_cond - et_uncond) * torch.norm(et_uncond)) +\
                (alpha_mix) * guidance_scale  * ((et_tgt_cond - et_uncond)/ torch.norm(et_tgt_cond - et_uncond) * torch.norm(et_uncond))
            zt = scheduler.step(et, t, zt)["prev_sample"]
            tvu.save_image((zt + 1) * 0.5, os.path.join(save_path, f'z{i}.png'))
            progress_bar.update(1)
    return zt
@torch.no_grad()
def run_ddim_mixture_p_sample_with_norm_gs(model, scheduler, zT, context,
                    NUM_DDIM_STEPS = 50,
                    num_inference_steps=50, 
                    start_time = 50, guidance_scale=7.5,
                    alpha_mix =0.2, 
                    save_dir= 'debug/original_null_inversion/', ):
    save_path = os.path.join(save_dir, 'generative_mixture')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    zt = zT.clone().detach()
    
    model.scheduler.set_timesteps(num_inference_steps)
    with tqdm(total=NUM_DDIM_STEPS, desc=f"DDIM generative process ") as progress_bar:
        for i, t in enumerate(model.scheduler.timesteps[-start_time:]):
            zt_cat = torch.cat([zt] * 3)
            noise_pred = model.unet(zt_cat, t, encoder_hidden_states=context.clone())["sample"]
            et_uncond, et_src_cond, et_tgt_cond = noise_pred.chunk(3)
            # et = et_cond 
            et = et_uncond + \
                (1 - alpha_mix) * guidance_scale *( (et_src_cond - et_uncond)/ torch.norm(et_src_cond - et_uncond) * torch.norm(et_uncond)) +\
                (alpha_mix) * guidance_scale  * ((et_tgt_cond - et_uncond)/ torch.norm(et_tgt_cond - et_uncond) * torch.norm(et_uncond))
            zt = model.scheduler.step(et, t, zt)["prev_sample"]
            tvu.save_image((zt + 1) * 0.5, os.path.join(save_path, f'z{i}.png'))
            progress_bar.update(1)
    return zt
@torch.no_grad()
def run_ddim_mixture_p_sample_with_norm_gs_p2p(model, scheduler, controller, zT, context,
                    NUM_DDIM_STEPS = 50,
                    num_inference_steps=50, 
                    start_time = 50, guidance_scale=7.5,
                    alpha_mix =0.2, 
                    save_dir= 'debug/original_null_inversion/', ):
    # ptp_utils.register_attention_control(model, controller)
    # save_path = os.path.join(save_dir, 'generative_mixture')
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    zt = zT.clone().detach()
    
    model.scheduler.set_timesteps(num_inference_steps)
    with tqdm(total=NUM_DDIM_STEPS, desc=f"DDIM generative process ") as progress_bar:
        for i, t in enumerate(model.scheduler.timesteps[-start_time:]):
            zt_cat = torch.cat([zt] * 3)
            noise_pred = model.unet(zt_cat, t, encoder_hidden_states=context.clone())["sample"]
            et_uncond, et_src_cond, et_tgt_cond = noise_pred.chunk(3)
            # et = et_cond 
            et = et_uncond + \
                (1 - alpha_mix) * guidance_scale *( (et_src_cond - et_uncond)/ 
                        torch.norm(et_src_cond - et_uncond) * torch.norm(et_uncond)) +\
                (alpha_mix) * guidance_scale  * ((et_tgt_cond - et_uncond)/ 
                        torch.norm(et_tgt_cond - et_uncond) * torch.norm(et_uncond))
            zt = model.scheduler.step(et, t, zt)["prev_sample"]
            # tvu.save_image((zt + 1) * 0.5, os.path.join(save_path, f'z{i}.png'))
            progress_bar.update(1)
    return zt
@torch.no_grad()
def run_ddim_mixture_p_sample(model, scheduler, zT, context,
                    NUM_DDIM_STEPS = 50,
                    num_inference_steps=50, 
                    start_time = 50, guidance_scale=7.5,
                    alpha_mix =0.2, 
                    save_dir= 'debug/original_null_inversion/', ):
    save_path = os.path.join(save_dir, 'generative_mixture')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    zt = zT.clone().detach()
    
    model.scheduler.set_timesteps(num_inference_steps)
    with tqdm(total=NUM_DDIM_STEPS, desc=f"DDIM generative process ") as progress_bar:
        for i, t in enumerate(model.scheduler.timesteps[-start_time:]):
            zt_cat = torch.cat([zt] * 3)
            noise_pred = model.unet(zt_cat, t, encoder_hidden_states=context.clone())["sample"]
            et_uncond, et_src_cond, et_tgt_cond = noise_pred.chunk(3)
            # et = et_cond 
            et = et_uncond + \
                (1 - alpha_mix) * guidance_scale * (et_src_cond - et_uncond) +\
                (alpha_mix) * guidance_scale  * (et_tgt_cond - et_uncond)
            zt = model.scheduler.step(et, t, zt)["prev_sample"]
            tvu.save_image((zt + 1) * 0.5, os.path.join(save_path, f'z{i}.png'))
            progress_bar.update(1)
    return zt
def run_ddim_p_sample_with_grad(model, scheduler, zT, context, 
                    NUM_DDIM_STEPS = 50,
                    num_inference_steps=50, 
                    start_time = 50, guidance_scale=7.5,
                    save_dir= 'debug/original_null_inversion/', ):
    save_path = os.path.join(save_dir, 'generative')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    zt = zT.clone().detach()
    model.scheduler.set_timesteps(num_inference_steps)
    with tqdm(total=NUM_DDIM_STEPS, desc=f"DDIM generative process ") as progress_bar:
        for i, t in enumerate(model.scheduler.timesteps[-start_time:]):
            zt_cat = torch.cat([zt] * 2)
            #--If not the final step, we use torch.no_grad()
            if (i +1) < 49:
                with torch.no_grad():
                    noise_pred = model.unet(zt_cat, t, encoder_hidden_states=context.clone())["sample"]
                    et_uncond, et_cond = noise_pred.chunk(2)
                    et = et_uncond + guidance_scale * (et_cond - et_uncond)
                    zt = model.scheduler.step(et, t, zt)["prev_sample"]
            else :
                with torch.enable_grad():
                    noise_pred = model.unet(zt_cat, t, encoder_hidden_states=context.clone())["sample"]
                    et_uncond, et_cond = noise_pred.chunk(2)
                    et = et_uncond + guidance_scale * (et_cond - et_uncond)
                    zt = model.scheduler.step(et, t, zt)["prev_sample"]
            tvu.save_image((zt + 1) * 0.5, os.path.join(save_path, f'z{i}.png'))
            progress_bar.update(1)
    return zt
@torch.no_grad()
def run_custom_p_sample_with_gs_p2p(model, scheduler, controller, zT, context, 
                    NUM_DDIM_STEPS = 50,
                    num_inference_steps=50, 
                    start_time = 50, guidance_scale=5.,
                    save_dir= 'debug/original_null_inversion/', ):
    ptp_utils.register_attention_control(model, controller)
    save_path = os.path.join(save_dir, 'generative')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    zt = zT.clone().detach()
    scheduler.set_timesteps(num_inference_steps)
    with tqdm(total=NUM_DDIM_STEPS, desc=f"Custom generative process ") as progress_bar:
        for i, t in enumerate(scheduler.timesteps[-start_time:]):
            zt_cat = torch.cat([zt] * 2)
            noise_pred = model.unet(zt_cat, t, encoder_hidden_states=context.clone())["sample"]
            et_uncond, et_cond = noise_pred.chunk(2)
            # et = et_cond 
            et = et_uncond + guidance_scale * (et_cond - et_uncond) /\
                torch.norm(et_cond - et_uncond) * torch.norm(et_uncond)
            zt = scheduler.step(et, t, zt)["prev_sample"]
            tvu.save_image((zt + 1) * 0.5, os.path.join(save_path, f'z{i}.png'))
            progress_bar.update(1)
    return zt

@torch.no_grad()
def run_ddim_p_sample_with_gs_p2p(model, scheduler, controller, zT, context, 
                    NUM_DDIM_STEPS = 50,
                    num_inference_steps=50, 
                    start_time = 50, guidance_scale=7.5,
                    save_dir= 'debug/original_null_inversion/', ):
    # ptp_utils.register_attention_control(model, controller)
    save_path = os.path.join(save_dir, 'generative')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    zt = zT.clone().detach()
    model.scheduler.set_timesteps(num_inference_steps)
    with tqdm(total=NUM_DDIM_STEPS, desc=f"DDIM generative process ") as progress_bar:
        for i, t in enumerate(model.scheduler.timesteps[-start_time:]):
            zt_cat = torch.cat([zt] * 2)
            noise_pred = model.unet(zt_cat, t, encoder_hidden_states=context.clone())["sample"]
            et_uncond, et_cond = noise_pred.chunk(2)
            et = et_uncond + guidance_scale * (et_cond - et_uncond)/\
                torch.norm(et_cond - et_uncond) * torch.norm(et_uncond)
            zt = model.scheduler.step(et, t, zt)["prev_sample"]
            tvu.save_image((zt + 1) * 0.5, os.path.join(save_path, f'z{i}.png'))
            progress_bar.update(1)
    return zt

@torch.no_grad()
def run_ddim_p_sample(model, scheduler, zT, context, 
                    NUM_DDIM_STEPS = 50,
                    num_inference_steps=50, 
                    start_time = 50, guidance_scale=7.5,
                    save_dir= 'debug/original_null_inversion/', ):
    save_path = os.path.join(save_dir, 'generative')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    zt = zT.clone().detach()
    model.scheduler.set_timesteps(num_inference_steps)
    with tqdm(total=NUM_DDIM_STEPS, desc=f"DDIM generative process ") as progress_bar:
        for i, t in enumerate(model.scheduler.timesteps[-start_time:]):
            zt_cat = torch.cat([zt] * 2)
            noise_pred = model.unet(zt_cat, t, encoder_hidden_states=context.clone())["sample"]
            et_uncond, et_cond = noise_pred.chunk(2)
            # et = et_cond 
            et = et_uncond + guidance_scale * (et_cond - et_uncond)
            zt = model.scheduler.step(et, t, zt)["prev_sample"]
            tvu.save_image((zt + 1) * 0.5, os.path.join(save_path, f'z{i}.png'))
            progress_bar.update(1)
    return zt

@torch.no_grad()
def run_ddim_p_sample_norm_gs(model, scheduler, zT, context, 
                    NUM_DDIM_STEPS = 50,
                    num_inference_steps=50, 
                    start_time = 50, guidance_scale=7.5,
                    save_dir= 'debug/original_null_inversion/', ):
    save_path = os.path.join(save_dir, 'generative')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    zt = zT.clone().detach()
    model.scheduler.set_timesteps(num_inference_steps)
    with tqdm(total=NUM_DDIM_STEPS, desc=f"DDIM generative process ") as progress_bar:
        for i, t in enumerate(model.scheduler.timesteps[-start_time:]):
            zt_cat = torch.cat([zt] * 2)
            noise_pred = model.unet(zt_cat, t, encoder_hidden_states=context.clone())["sample"]
            et_uncond, et_cond = noise_pred.chunk(2)
            # et = et_cond 
            et = et_uncond + guidance_scale * (et_cond - et_uncond) /\
                torch.norm(et_cond - et_uncond) * torch.norm(et_uncond)
            zt = model.scheduler.step(et, t, zt)["prev_sample"]
            # tvu.save_image((zt + 1) * 0.5, os.path.join(save_path, f'z{i}.png'))
            progress_bar.update(1)
    return zt

@torch.no_grad()
def run_ddim_p_sample_p2p(model, scheduler, zT, context, controller, 
                    NUM_DDIM_STEPS = 50,
                    num_inference_steps=50, 
                    start_time = 50, guidance_scale=7.5,
                    save_dir= 'debug/original_null_inversion/',):
    ptp_utils.register_attention_control(model, controller)
    save_path = os.path.join(save_dir, 'generative')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    zt = zT.clone().detach()
    model.scheduler.set_timesteps(num_inference_steps)
    with tqdm(total=NUM_DDIM_STEPS, desc=f"DDIM generative process ") as progress_bar:
        for i, t in enumerate(model.scheduler.timesteps[-start_time:]):
            zt_cat = torch.cat([zt] * 2)
            noise_pred = model.unet(zt_cat, t, encoder_hidden_states=context.clone())["sample"]
            et_uncond, et_cond = noise_pred.chunk(2)
            # et = et_cond 
            et = et_uncond + guidance_scale * (et_cond - et_uncond)
            zt = model.scheduler.step(et, t, zt)["prev_sample"]
            tvu.save_image((zt + 1) * 0.5, os.path.join(save_path, f'z{i}.png'))
            progress_bar.update(1)
    return zt

@torch.no_grad()
def run_custom_mixture_p_sample_with_norm_gs_p2p_hybrid(model, scheduler, controller, zT, context,
                    NUM_DDIM_STEPS = 50,
                    num_inference_steps=50, 
                    start_time = 50, guidance_scale=7.5,
                    alpha_mix =0.2,
                    start_idx_mix = 10, 
                    save_dir= 'debug/original_null_inversion/', ):
    ptp_utils.register_attention_control(model, controller)
    save_path = os.path.join(save_dir, 'generative_hybrid_mix')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    zt = zT.clone().detach()
    scheduler.set_timesteps(num_inference_steps)

    with tqdm(total=NUM_DDIM_STEPS, desc=f"Custom generative process ") as progress_bar:
        for i, t in enumerate(scheduler.timesteps[-start_time:]):
            zt_cat = torch.cat([zt] * 3)
            noise_pred = model.unet(zt_cat, t, encoder_hidden_states=context.clone())["sample"]
            et_uncond, et_src_cond, et_tgt_cond = noise_pred.chunk(3)
            # et = et_cond 
            if i < start_idx_mix:
                et = et_uncond + guidance_scale * ((et_tgt_cond - et_uncond)/ torch.norm(et_tgt_cond - et_uncond) * torch.norm(et_uncond))
            else:
                et = et_uncond + \
                    (1 - alpha_mix) * guidance_scale *( (et_src_cond - et_uncond)/ 
                                        torch.norm(et_src_cond - et_uncond) * torch.norm(et_uncond)) +\
                    (alpha_mix) * guidance_scale  * ((et_tgt_cond - et_uncond)/ 
                                        torch.norm(et_tgt_cond - et_uncond) * torch.norm(et_uncond))
            zt = scheduler.step(et, t, zt)["prev_sample"]
            tvu.save_image((zt + 1) * 0.5, os.path.join(save_path, f'z{i}.png'))
            progress_bar.update(1)
    return zt
@torch.no_grad()
def run_ddim_mixture_p_sample_with_norm_gs_p2p_hybrid(model, scheduler, controller, zT, context,
                    NUM_DDIM_STEPS = 50,
                    num_inference_steps=50, 
                    start_time = 50, guidance_scale=7.5,
                    alpha_mix =0.2, 
                    start_idx_mix = 10,
                    save_dir= 'debug/original_null_inversion/', ):
    ptp_utils.register_attention_control(model, controller)
    save_path = os.path.join(save_dir, 'generative_hybrid_mix')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    zt = zT.clone().detach()
    
    model.scheduler.set_timesteps(num_inference_steps)
    with tqdm(total=NUM_DDIM_STEPS, desc=f"DDIM generative process ") as progress_bar:
        for i, t in enumerate(model.scheduler.timesteps[-start_time:]):
            zt_cat = torch.cat([zt] * 3)
            noise_pred = model.unet(zt_cat, t, encoder_hidden_states=context.clone())["sample"]
            et_uncond, et_src_cond, et_tgt_cond = noise_pred.chunk(3)
            if i < start_idx_mix:
                et = et_uncond + guidance_scale * ((et_tgt_cond - et_uncond)/ torch.norm(et_tgt_cond - et_uncond) * torch.norm(et_uncond))
            else:
                et = et_uncond + \
                    (1 - alpha_mix) * guidance_scale *( (et_src_cond - et_uncond)/ torch.norm(et_src_cond - et_uncond) * torch.norm(et_uncond)) +\
                    (alpha_mix) * guidance_scale  * ((et_tgt_cond - et_uncond)/ torch.norm(et_tgt_cond - et_uncond) * torch.norm(et_uncond))
            zt = model.scheduler.step(et, t, zt)["prev_sample"]
            tvu.save_image((zt + 1) * 0.5, os.path.join(save_path, f'z{i}.png'))
            progress_bar.update(1)
    return zt

@torch.no_grad()
def run_ddim_p_sample_norm_gs_paper(model, zT, context,
                    NUM_DDIM_STEPS = 50,
                    num_inference_steps=50, 
                    start_time = 50, guidance_scale=7.5):
    
    zt = zT.clone().detach()
    model.scheduler.set_timesteps(num_inference_steps)
    with tqdm(total=NUM_DDIM_STEPS, desc=f"DDIM generative process ") as progress_bar:
        for i, t in enumerate(model.scheduler.timesteps[-start_time:]):
            zt_cat = torch.cat([zt] * 2)
            noise_pred = model.unet(zt_cat, t, encoder_hidden_states=context.clone())["sample"]
            et_uncond, et_cond = noise_pred.chunk(2)
            # et = et_cond 
            et = et_uncond + \
                guidance_scale *( (et_cond - et_uncond)/ 
                torch.norm(et_cond - et_uncond) * torch.norm(et_uncond))
            
            zt = model.scheduler.step(et, t, zt)["prev_sample"]
            # tvu.save_image((zt + 1) * 0.5,  f'test/ddim_z{i}.png')
            progress_bar.update(1)
    return zt
@torch.no_grad()
def run_pndm_p_sample_norm_gs_paper(model, zT, context, scheduler,
                    NUM_DDIM_STEPS = 50,
                    num_inference_steps=50, 
                    start_time = 50, guidance_scale=7.5):
    
    zt = zT.clone().detach()
    scheduler.set_timesteps(num_inference_steps)
    with tqdm(total=NUM_DDIM_STEPS, desc=f"PNDM generative process ") as progress_bar:
        for i, t in enumerate(scheduler.timesteps[-start_time:]):
            zt_cat = torch.cat([zt] * 2)
            noise_pred = model.unet(zt_cat, t, encoder_hidden_states=context.clone())["sample"]
            et_uncond, et_cond = noise_pred.chunk(2)
            # et = et_cond 
            et = et_uncond + \
                guidance_scale *( (et_cond - et_uncond)/ 
                torch.norm(et_cond - et_uncond) * torch.norm(et_uncond))
            
            zt = scheduler.step(et, t, zt)["prev_sample"]
            # tvu.save_image((zt + 1) * 0.5,  f'test/pndm_z{i}.png')
            progress_bar.update(1)
    return zt
