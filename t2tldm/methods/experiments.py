import os 
from loguru import logger
import torch
import torchvision.utils as tvu

from utils import img_utils, diffusion_utils, prompt_utils, denoising_utils, constants
from methods.optimization import optimize_emb_with_text_recog_loss
from baselines.imagic.method import run_optimize_emb, run_finetune_et_model, run_finetune_et_model_with_text_recog, run_optimize_emb_augm
from baselines.prompt2prompt.method import AttentionStore, show_cross_attention
def run_protocol_1(ldm_stable, scheduler, z0, zT, 
                    uncond_emb, src_cond_emb, 
                    tgt_cond_emb, 
                    target_text,
                    yaml_file, 
                    step_recog_loss = 35,
                    NUM_DDIM_STEPS = 50, 
                    save_dir='debug/finetune', 
                    device = 'cuda'):
    logger.info(f'Running protocol 1')
    opt_src_emb = run_optimize_emb(ldm_stable, scheduler, z0, src_cond_emb, device=device)
    context = torch.cat([uncond_emb, opt_src_emb])
    z0 = denoising_utils.run_ddim_p_sample(ldm_stable, scheduler, zT, context, 
                        NUM_DDIM_STEPS, save_dir=save_dir)
    x0 =img_utils.latent2im(ldm_stable, z0)
    tvu.save_image((x0 + 1) * 0.5, os.path.join(save_dir, f'p1_x0_rec.png'))

def run_protocol_2(ldm_stable, scheduler, z0, zT, 
                    uncond_emb, src_cond_emb, 
                    tgt_cond_emb, 
                    target_text,
                    yaml_file, 
                    step_recog_loss = 35,
                    NUM_DDIM_STEPS = 50, 
                    save_dir='debug/finetune', 
                    device = 'cuda'):
    # info = constants.protocol_details['p1']['info']
    logger.info(f'Running protocol 2')
    opt_src_emb = run_optimize_emb(ldm_stable, scheduler, z0, src_cond_emb, device=device)
    run_finetune_et_model(ldm_stable, scheduler, z0, opt_src_emb, device=device)

    context = torch.cat([uncond_emb, opt_src_emb])
    z0 = denoising_utils.run_ddim_p_sample(ldm_stable, scheduler, zT, context, 
                        NUM_DDIM_STEPS, save_dir=save_dir)
    x0 =img_utils.latent2im(ldm_stable, z0)
    tvu.save_image((x0 + 1) * 0.5, os.path.join(save_dir, f'p2_x0_rec.png'))

def run_protocol_3(ldm_stable, scheduler, z0, zT, 
                    uncond_emb, src_cond_emb, 
                    tgt_cond_emb, 
                    target_text,
                    yaml_file, 
                    step_recog_loss = 35,
                    NUM_DDIM_STEPS = 50, 
                    num_opt_emb_iter=500, 
                    num_ft_et_iter =1500,
                    save_dir='debug/finetune', 
                    device = 'cuda'):
    logger.info(f'Running protocol 3')

    opt_src_emb = run_optimize_emb(ldm_stable, scheduler, z0, src_cond_emb, 
                    num_iter=num_opt_emb_iter,device=device)

    # opt_tgt_emb = run_optimize_emb(ldm_stable, scheduler, z0, tgt_cond_emb, 
    #                 num_iter=num_opt_emb_iter,device=device)

    run_finetune_et_model(ldm_stable, scheduler, z0, opt_src_emb, 
                            num_iter=num_ft_et_iter,device=device)

    context = torch.cat([uncond_emb, opt_src_emb, tgt_cond_emb]) 
    # context = torch.cat([uncond_emb, opt_src_emb, tgt_cond_emb]) 

    save_sampling_dir = os.path.join(save_dir, f'opt_emb_iter{str(num_opt_emb_iter)}', f'ft_et_iter{str(num_ft_et_iter)}')
    if not os.path.exists(save_sampling_dir):
        os.makedirs(save_sampling_dir)
    for gs in [0.1, 0.125, 0.15, 0.175, 0.2, 0.3, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7]:
        curr_zT = zT.clone()
        for alpha in (0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 1, 1.1):
            z0 = denoising_utils.run_ddim_mixture_p_sample_with_norm_gs(ldm_stable, scheduler, curr_zT, context, 
                        NUM_DDIM_STEPS, guidance_scale=gs, alpha_mix= alpha, save_dir=save_dir)
            x0 =img_utils.latent2im(ldm_stable, z0)
            tvu.save_image((x0 + 1) * 0.5, os.path.join(save_sampling_dir, f'[gs_{str(gs)}][alp_{str(alpha)}]_x0_rec.jpg'))

    # z0 = denoising_utils.run_ddim_mixture_p_sample(ldm_stable, scheduler, zT, context, 
    #                     NUM_DDIM_STEPS, guidance_scale=5., alpha_mix= .4, save_dir=save_dir)
def run_debug_protocol_4(ldm_stable, scheduler, z0, zT, 
                    uncond_emb, src_cond_emb, 
                    tgt_cond_emb, 
                    target_text,
                    yaml_file, 
                    step_recog_loss = 35,
                    NUM_DDIM_STEPS = 50, 
                    num_opt_emb_iter=500, 
                    num_ft_et_iter =1500,
                    update_zT=False,
                    save_dir='debug/finetune', 
                    device = 'cuda'):
    logger.info(f'Running protocol 4')
    
    
    opt_src_emb = run_optimize_emb(ldm_stable, scheduler, z0, src_cond_emb, 
                                num_iter=num_opt_emb_iter,device=device)

    run_finetune_et_model(ldm_stable, scheduler, z0, opt_src_emb, 
                            num_iter=num_ft_et_iter,device=device)

    opt_tgt_emb = optimize_emb_with_text_recog_loss(ldm_stable, scheduler, zT, 
                                uncond_emb, tgt_cond_emb, 
                                target_text= target_text,
                                yaml_file= yaml_file,
                                guidance_scale=0.2,
                                step_recog_loss = step_recog_loss,
                                save_dir=save_dir)

    context = torch.cat([uncond_emb, opt_src_emb, opt_tgt_emb]) 

    #--Update new zT with finetuned ET
    if update_zT:
        all_zt, zT = diffusion_utils.run_ddim_q_sample(ldm_stable, scheduler, z0.clone(), torch.cat([uncond_emb, src_cond_emb.clone()]), 
                    NUM_DDIM_STEPS, COND_Q_SAMPLE=True, save_dir=save_dir)


    save_sampling_dir = os.path.join(save_dir, 'mixture', 'p4', f'opt_emb_iter{str(num_opt_emb_iter)}', f'ft_et_iter{str(num_ft_et_iter)}')
    if not os.path.exists(save_sampling_dir):
        os.makedirs(save_sampling_dir)
    for gs in [0.1, 0.125, 0.15, 0.175, 0.2, 0.3, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7]:
        curr_zT = zT.clone()
        for alpha in (0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 1, 1.1):
            z0 = denoising_utils.run_ddim_mixture_p_sample_with_norm_gs(ldm_stable, scheduler, curr_zT, context, 
                        NUM_DDIM_STEPS, guidance_scale=gs, alpha_mix= alpha, save_dir=save_dir)
            x0 =img_utils.latent2im(ldm_stable, z0)
            tvu.save_image((x0 + 1) * 0.5, os.path.join(save_sampling_dir, f'[gs_{str(gs)}][alp_{str(alpha)}]_x0_rec.jpg'))
    # z0 = denoising_utils.run_ddim_mixture_p_sample(ldm_stable, scheduler, zT, context, 
    #                     NUM_DDIM_STEPS, guidance_scale=5., alpha_mix= .9, save_dir=save_dir)
    # x0 =img_utils.latent2im(ldm_stable, z0)
    # tvu.save_image((x0 + 1) * 0.5, os.path.join(save_dir, f'p4_x0_rec.png'))
def run_debug_protocol_7(ldm_stable, scheduler, z0, zT, 
                    uncond_emb, src_cond_emb, 
                    tgt_cond_emb, 
                    target_text,
                    yaml_file, 
                    step_recog_loss = 35,
                    NUM_DDIM_STEPS = 50, 
                    num_opt_emb_iter=500, 
                    num_ft_et_iter =1500,
                    ft_gs = 0.5,
                    update_zT=False,
                    scheduler_name ='ddim',
                    save_dir='debug/finetune', 
                    device = 'cuda'):
    logger.info(f'Running protocol 4')
    
    #--Step 1: Optimize target embedding using text recognition loss
    # opt_tgt_emb = optimize_emb_with_text_recog_loss(ldm_stable, scheduler, zT, 
    #                             uncond_emb, tgt_cond_emb, 
    #                             target_text= target_text,
    #                             yaml_file= yaml_file, 
    #                             guidance_scale=ft_gs,
    #                             step_recog_loss = step_recog_loss,
    #                             save_dir=save_dir)

    #--Step 2: Optimize source embedding using MSE loss
    opt_src_emb = run_optimize_emb(ldm_stable, scheduler, z0, src_cond_emb, 
                                num_iter=num_opt_emb_iter,device=device)
    #--Step 3: Finetune et model using MSE loss
    # run_finetune_et_model(ldm_stable, scheduler, z0, opt_src_emb, 
    #                         num_iter=num_ft_et_iter,device=device)

    

    # context = torch.cat([uncond_emb, opt_src_emb, opt_tgt_emb]) 
    context = torch.cat([uncond_emb, opt_src_emb, tgt_cond_emb.detach()]) 
    #--Step 4: Refine et model with text recognition loss
    # run_finetune_et_model_with_text_recog(ldm_stable, scheduler, zT,
    #                 uncond_emb, tgt_cond_emb.detach(), 
    #                 target_text= target_text,
    #                 yaml_file= yaml_file, 
    #                 num_iter= int(100 * ft_gs),
    #                 guidance_scale=ft_gs,
    #                 step_recog_loss = step_recog_loss,
    #                 save_dir=save_dir)

    #--Step 4: Refine et model with text recognition loss
    # run_finetune_et_model_with_text_recog(ldm_stable, scheduler, zT,
    #                 uncond_emb, opt_tgt_emb, 
    #                 target_text= target_text,
    #                 yaml_file= yaml_file, 
    #                 num_iter= int(100 * ft_gs),
    #                 guidance_scale=ft_gs,
    #                 step_recog_loss = step_recog_loss,
    #                 save_dir=save_dir)

    save_sampling_dir = os.path.join(save_dir, 'p4', scheduler_name, 
                                    f'opt_emb_iter{str(num_opt_emb_iter)}', 
                                    f'ft_et_iter{str(num_ft_et_iter)}')

    if not os.path.exists(save_sampling_dir):
        os.makedirs(save_sampling_dir)
    for gs in [0.2, 0.3, 0.4, 0.5 ]:
    # for gs in [0.1, 0.125, 0.15, 0.175, 0.2, 0.3, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7]:
        curr_zT = zT.clone()
        for alpha in (0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 1, 1.1):
            if scheduler_name!= 'ddim':
                logger.info(f'Using {scheduler_name} scheduler')
                z0 = denoising_utils.run_custom_mixture_p_sample_with_norm_gs(ldm_stable, scheduler, curr_zT, context, 
                            NUM_DDIM_STEPS, guidance_scale=gs, alpha_mix= alpha, save_dir=save_dir)
            else:
                logger.info(f'Using {scheduler_name} scheduler')
                z0 = denoising_utils.run_ddim_mixture_p_sample_with_norm_gs(ldm_stable, scheduler, curr_zT, context, 
                            NUM_DDIM_STEPS, guidance_scale=gs, alpha_mix= alpha, save_dir=save_dir)
            x0 =img_utils.latent2im(ldm_stable, z0)
            tvu.save_image((x0 + 1) * 0.5, os.path.join(save_sampling_dir, f'[gs_{str(gs)}][alp_{str(alpha)}]_x0_rec.jpg'))
def run_protocol_4(ldm_stable, scheduler, z0, zT, 
                    uncond_emb, src_cond_emb, 
                    tgt_cond_emb, 
                    target_text,
                    yaml_file, 
                    step_recog_loss = 35,
                    NUM_DDIM_STEPS = 50, 
                    save_dir='debug/finetune', 
                    device = 'cuda'):
    logger.info(f'Running protocol 4')

    opt_tgt_emb = optimize_emb_with_text_recog_loss(ldm_stable, scheduler, zT, uncond_emb, tgt_cond_emb, 
                                target_text= target_text,
                                yaml_file= yaml_file, 
                                step_recog_loss = step_recog_loss,
                                save_dir=save_dir)

    opt_src_emb = run_optimize_emb(ldm_stable, scheduler, z0, src_cond_emb, device=device)
    run_finetune_et_model(ldm_stable, scheduler, z0, opt_src_emb, device=device)
    context = torch.cat([uncond_emb, opt_src_emb, opt_tgt_emb]) 

   
    z0 = denoising_utils.run_ddim_mixture_p_sample(ldm_stable, scheduler, zT, context, 
                        NUM_DDIM_STEPS, guidance_scale=5., alpha_mix= .9, save_dir=save_dir)
    x0 =img_utils.latent2im(ldm_stable, z0)
    tvu.save_image((x0 + 1) * 0.5, os.path.join(save_dir, f'p4_x0_rec.png'))

def run_protocol_5(ldm_stable, scheduler, z0, zT, 
                    uncond_emb, src_cond_emb, 
                    tgt_cond_emb, 
                    target_text,
                    yaml_file, 
                    step_recog_loss = 35,
                    NUM_DDIM_STEPS = 50, 
                    save_dir='debug/finetune', 
                    device = 'cuda'):
    logger.info(f'Running protocol 5')
    opt_src_emb = run_optimize_emb(ldm_stable, scheduler, z0, src_cond_emb, device=device)
    run_finetune_et_model(ldm_stable, scheduler, z0, opt_src_emb, device=device)
    opt_tgt_emb = optimize_emb_with_text_recog_loss(ldm_stable, scheduler, zT, uncond_emb, tgt_cond_emb, 
                                target_text= target_text,
                                yaml_file= yaml_file, 
                                step_recog_loss = step_recog_loss,
                                save_dir=save_dir)
    
    context = torch.cat([uncond_emb, opt_src_emb, opt_tgt_emb]) 

    z0 = denoising_utils.run_ddim_mixture_p_sample(ldm_stable, scheduler, zT, context, 
                        NUM_DDIM_STEPS, guidance_scale=5., alpha_mix= .9, save_dir=save_dir)
    x0 =img_utils.latent2im(ldm_stable, z0)
    tvu.save_image((x0 + 1) * 0.5, os.path.join(save_dir, f'p5_x0_rec.png'))
def run_protocol_6(ldm_stable, scheduler, z0, zT, 
                    uncond_emb, src_cond_emb, 
                    tgt_cond_emb, 
                    target_text,
                    yaml_file, 
                    step_recog_loss = 35,
                    NUM_DDIM_STEPS = 50, 
                    save_dir='debug/finetune', 
                    device = 'cuda'):
    logger.info(f'Running protocol 6')
    opt_src_emb = run_optimize_emb(ldm_stable, scheduler, z0, src_cond_emb, device=device)
    run_finetune_et_model(ldm_stable, scheduler, z0, opt_src_emb, device=device)
    opt_tgt_emb = optimize_emb_with_text_recog_loss(ldm_stable, scheduler, zT, uncond_emb, tgt_cond_emb, 
                                target_text= target_text,
                                yaml_file= yaml_file, 
                                step_recog_loss = 35,
                                save_dir=save_dir)
    
    context = torch.cat([uncond_emb, opt_src_emb, opt_tgt_emb]) 

    z0 = denoising_utils.run_ddim_mixture_p_sample(ldm_stable, scheduler, zT, context, 
                        NUM_DDIM_STEPS, guidance_scale=5., alpha_mix= .9, save_dir=save_dir)
    x0 =img_utils.latent2im(ldm_stable, z0)
    tvu.save_image((x0 + 1) * 0.5, os.path.join(save_dir, f'p6_x0_rec.png'))

def run_p2p(ldm_stable, scheduler, tokenizer, controller, z0, zT,
            src_prompt, tgt_prompts, target_text,
            yaml_file,
            step_recog_loss =35,
            OPT_SRC_EMB = False, 
            OPT_TGT_EMB =False,
            FT_ET = False,
            save_dir= 'debug/finetune',
            MAX_NUM_WORDS=77, NUM_DDIM_STEPS=50, 
            device='cuda'):
    save_path = os.path.join(save_dir, 'prompt2prompt')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    uncond_emb, src_cond_emb = prompt_utils.gen_init_prompt_to_emb(ldm_stable, tokenizer, 
                            src_prompt, MAX_NUM_WORDS, device=device)
    _, tgt_cond_emb = prompt_utils.gen_init_prompt_to_emb(ldm_stable, tokenizer, 
                            tgt_prompts, MAX_NUM_WORDS, device=device)
    savename = ''
    if OPT_SRC_EMB:
        logger.debug('With finetuning source embedding')
        opt_src_emb = run_optimize_emb(ldm_stable, scheduler, z0, src_cond_emb, device=device)
        savename += '[opt_src]'
    else:
        logger.debug('Without finetuning source embedding')
        opt_src_emb = src_cond_emb.detach().clone()

    if OPT_TGT_EMB:
        logger.debug('With finetuning target embedding')
        opt_tgt_emb = optimize_emb_with_text_recog_loss(ldm_stable, scheduler, zT, uncond_emb, tgt_cond_emb, 
                                target_text= target_text,
                                yaml_file= yaml_file, 
                                step_recog_loss = step_recog_loss,
                                save_dir=save_path)
        savename += '[opt_tgt]'
    else:
        logger.debug('Without finetuning target embedding')
        opt_tgt_emb = tgt_cond_emb    
    if FT_ET:
        logger.debug('With finetuning Et model')
        run_finetune_et_model(ldm_stable, scheduler, z0, opt_src_emb, device=device)
        savename += '[ft_et]'
    else:
        pass
    batch_cond_emb = torch.cat([opt_src_emb, opt_tgt_emb])
    batch_uncond_emb = torch.cat([uncond_emb] * batch_cond_emb.shape[0])
    
    context = torch.cat([batch_uncond_emb, batch_cond_emb])
    #--Sampling
    batch_zT = torch.cat([zT] * batch_cond_emb.shape[0])
    
    z0 = denoising_utils.run_ddim_p_sample_p2p(ldm_stable, scheduler, batch_zT, context, controller,
                        NUM_DDIM_STEPS, guidance_scale=5., save_dir=save_dir)
    x0 =img_utils.latent2im(ldm_stable, z0)
    tvu.save_image((x0 + 1) * 0.5, os.path.join(save_path, f'{savename}x0_rec.png'))

def run_imagic(ldm_stable, scheduler, tokenizer, z0, zT,
            src_prompt, tgt_prompts, target_text,
            yaml_file,
            step_recog_loss =35,
            OPT_SRC_EMB = False, 
            OPT_TGT_EMB =False,
            FT_ET = False,
            RANDOM_NOISE = True,
            save_dir= 'debug/finetune',
            MAX_NUM_WORDS=77, 
            NUM_DDIM_STEPS=50,
            num_opt_emb_iter=500, 
            num_ft_et_iter =1000,
            with_norm_gs=False,
            with_seq_gs=False,
            device='cuda'):
    save_path = os.path.join(save_dir, 'imagic')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    src_prompt = tgt_prompts
    uncond_emb, src_cond_emb = prompt_utils.gen_init_prompt_to_emb(ldm_stable, tokenizer, 
                            src_prompt, MAX_NUM_WORDS, device=device)
    _, tgt_cond_emb = prompt_utils.gen_init_prompt_to_emb(ldm_stable, tokenizer, 
                            tgt_prompts, MAX_NUM_WORDS, device=device)
    savename = ''
    if OPT_SRC_EMB:
        logger.debug('With finetuning source embedding')
        opt_src_emb = run_optimize_emb(ldm_stable, scheduler, z0, src_cond_emb, 
                            # num_iter= 425, device=device)
                            num_iter= num_opt_emb_iter, device=device)
        savename += '[opt_src]'
    else:
        logger.debug('Without finetuning source embedding')
        opt_src_emb = src_cond_emb.detach().clone()

    if OPT_TGT_EMB:
        logger.debug('With finetuning target embedding')
        opt_tgt_emb = optimize_emb_with_text_recog_loss(ldm_stable, scheduler, zT, uncond_emb, tgt_cond_emb, 
                                target_text= target_text,
                                yaml_file= yaml_file, 
                                step_recog_loss = step_recog_loss,
                                save_dir=save_path)
        savename += '[opt_tgt]'
    else:
        logger.debug('Without finetuning target embedding')
        opt_tgt_emb = tgt_cond_emb.detach().clone()
    if FT_ET:
        logger.debug('With finetuning Et model')
        run_finetune_et_model(ldm_stable, scheduler, z0, opt_src_emb, 
                    num_iter=num_ft_et_iter,device=device)
                    # num_iter=1500,device=device)
        savename += '[ft_et]'
    else:
        pass
    #--With zT from random noise
    if RANDOM_NOISE:
        zT = torch.randn_like(zT)
        savename += '[rand_noise]'
    #--Save all sapling ablation for specific opt_emb_iter and ft_et_iter
    save_sampling_dir = os.path.join(save_path, f'opt_emb_iter{str(num_opt_emb_iter)}', f'ft_et_iter{str(num_ft_et_iter)}')
    if not os.path.exists(save_sampling_dir):
        os.makedirs(save_sampling_dir)
    # orig_zT = zT.clone()
    if with_seq_gs:
        for gs in [0.1, 0.125, 0.15, 0.175, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        # for gs in [0.1, 0.125, 0.15, 0.175, 0.2, 0.3, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7]:
            curr_zT = zT.clone()
            for alpha in (0.5, 0.55, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 1, 1.1):
                new_emb = alpha*src_cond_emb + (1-alpha)*opt_src_emb
                context = torch.cat([uncond_emb, new_emb])
                if with_norm_gs:
                    z0 = denoising_utils.run_ddim_p_sample_norm_gs(ldm_stable, scheduler, curr_zT, context,
                                    NUM_DDIM_STEPS, guidance_scale=gs, save_dir=save_dir)
                x0 =img_utils.latent2im(ldm_stable, z0)
                tvu.save_image((x0 + 1) * 0.5, os.path.join(save_sampling_dir, f'{savename}[gs_{str(gs)}][alp_{str(alpha)}]_x0_rec.jpg'))
    else:
        for alpha in (0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 1, 1.1):
        # for alpha in (0.8, 0.9, 1, 1.1):
            new_emb = alpha*src_cond_emb + (1-alpha)*opt_src_emb

            context = torch.cat([uncond_emb, new_emb])
            #--Sampling

            # z0 = denoising_utils.run_ddim_p_sample(ldm_stable, scheduler, zT, context,
            #                     NUM_DDIM_STEPS, guidance_scale=5., save_dir=save_dir)
            if with_norm_gs:
                z0 = denoising_utils.run_ddim_p_sample_norm_gs(ldm_stable, scheduler, zT, context,
                                NUM_DDIM_STEPS, guidance_scale=0.5, save_dir=save_dir)
            else:
                z0 = denoising_utils.run_ddim_p_sample(ldm_stable, scheduler, zT, context,
                                    NUM_DDIM_STEPS, guidance_scale=5., save_dir=save_dir)
            x0 =img_utils.latent2im(ldm_stable, z0)
            tvu.save_image((x0 + 1) * 0.5, os.path.join(save_sampling_dir, f'{savename}x0_rec_{str(alpha)}.png'))

def run_protocol_8(ldm_stable, tokenizer, scheduler, 
                    z0, zT, src_prp, tgt_prp,
                    uncond_emb, src_cond_emb, 
                    tgt_cond_emb, 
                    target_text,
                    yaml_file, 
                    controller_type='attn_store',
                    step_recog_loss = 35,
                    NUM_DDIM_STEPS = 50, 
                    num_opt_emb_iter=500, 
                    num_ft_et_iter =1500,
                    update_zT=False,
                    ft_gs=0.3,
                    exp_code = 'original_p2p',
                    scheduler_name='ddim',
                    save_dir='debug/finetune', 
                    device = 'cuda'):
    logger.info(f'Running protocol 8')
    save_sampling_dir = os.path.join(save_dir, 'p8', scheduler_name)
    if controller_type=='attn_store':
        attn = AttentionStore()
    if exp_code =='original_p2p':
        context = torch.cat([uncond_emb, src_cond_emb.detach()])
        if scheduler_name!= 'ddim':
            z0 = denoising_utils.run_ddim_p_sample_with_gs_p2p(ldm_stable, scheduler, attn, 
                                zT, context, guidance_scale=0.3, save_dir=save_sampling_dir)
        else:
            z0 = denoising_utils.run_custom_p_sample_with_gs_p2p(ldm_stable, scheduler, attn, 
                                zT, context, guidance_scale=0.3, save_dir=save_sampling_dir)
    elif exp_code =='p2p_opt_src':
        opt_src_emb = run_optimize_emb(ldm_stable, scheduler, z0, src_cond_emb, 
                                num_iter=num_opt_emb_iter, device=device)
        context = torch.cat([uncond_emb, opt_src_emb])
        z0 = denoising_utils.run_ddim_p_sample_with_gs_p2p(ldm_stable, scheduler, attn, 
                            zT, context, guidance_scale=0.3, save_dir=save_sampling_dir)
    elif exp_code =='p2p_opt_src_text_recog':
        opt_src_emb = optimize_emb_with_text_recog_loss(ldm_stable, scheduler, zT, 
                                uncond_emb, src_cond_emb, 
                                target_text= target_text,
                                yaml_file= yaml_file, 
                                guidance_scale=ft_gs,
                                step_recog_loss = step_recog_loss,
                                save_dir=save_dir)
        context = torch.cat([uncond_emb, opt_src_emb])
        z0 = denoising_utils.run_ddim_p_sample_with_gs_p2p(ldm_stable, scheduler, attn, 
                            zT, context, guidance_scale=0.3, save_dir=save_sampling_dir)
    elif exp_code =='p2p_ft_et':
        run_finetune_et_model(ldm_stable, scheduler, z0, src_cond_emb.detach(), 
                            num_iter=num_ft_et_iter,device=device)

        context = torch.cat([uncond_emb, src_cond_emb.detach()])
        z0 = denoising_utils.run_ddim_p_sample_with_gs_p2p(ldm_stable, scheduler, attn, 
                            zT, context, guidance_scale=0.3, save_dir=save_sampling_dir)
    elif exp_code =='p2p_opt_src_ft_et':
        
        opt_src_emb = run_optimize_emb(ldm_stable, scheduler, z0, src_cond_emb, 
                                num_iter=num_opt_emb_iter, device=device)
        
        run_finetune_et_model(ldm_stable, scheduler, z0, opt_src_emb, 
                            num_iter=num_ft_et_iter,device=device)

        context = torch.cat([uncond_emb, opt_src_emb])
        z0 = denoising_utils.run_ddim_p_sample_with_gs_p2p(ldm_stable, scheduler, attn, 
                            zT, context, guidance_scale=0.3, save_dir=save_sampling_dir)

    elif exp_code =='p2p_opt_src_text_recog_ft_et':
        opt_src_emb = optimize_emb_with_text_recog_loss(ldm_stable, scheduler, zT, 
                                uncond_emb, src_cond_emb, 
                                target_text= target_text,
                                yaml_file= yaml_file, 
                                guidance_scale=ft_gs,
                                step_recog_loss = step_recog_loss,
                                save_dir=save_dir)
        
        run_finetune_et_model(ldm_stable, scheduler, z0, opt_src_emb, 
                            num_iter=num_ft_et_iter,device=device)

        context = torch.cat([uncond_emb, opt_src_emb])
        z0 = denoising_utils.run_ddim_p_sample_with_gs_p2p(ldm_stable, scheduler, attn, 
                            zT, context, guidance_scale=0.3, save_dir=save_sampling_dir)
    elif exp_code =='p2p_ft_et_opt_src_text_recog':
        run_finetune_et_model(ldm_stable, scheduler, z0, src_cond_emb.detach(), 
                            num_iter=num_ft_et_iter,device=device)
        opt_src_emb = optimize_emb_with_text_recog_loss(ldm_stable, scheduler, zT, 
                                uncond_emb, src_cond_emb, 
                                target_text= target_text,
                                yaml_file= yaml_file, 
                                num_iter = int(100 * ft_gs),
                                guidance_scale=ft_gs,
                                step_recog_loss = step_recog_loss,
                                save_dir=save_dir)
        context = torch.cat([uncond_emb, opt_src_emb])
        z0 = denoising_utils.run_ddim_p_sample_with_gs_p2p(ldm_stable, scheduler, attn, 
                            zT, context, guidance_scale=0.3, save_dir=save_sampling_dir)
    x0 =img_utils.latent2im(ldm_stable, z0)
    tvu.save_image((x0 + 1) * 0.5, os.path.join(save_sampling_dir, f'[{exp_code}]_x0_rec.jpg'))
    #--Show cross attention
    save_attn_dir = os.path.join(save_sampling_dir, 'attn')
    if not os.path.exists(save_attn_dir):
        os.makedirs(save_attn_dir)
    show_cross_attention(tokenizer, src_prp, attn, res=16, from_where=["up", "down"], 
                        save_dir=save_attn_dir, save_name=f'[{exp_code}]')

def run_protocol_9(ldm_stable, tokenizer, scheduler, 
                    z0, zT, src_prp, tgt_prp,
                    uncond_emb, src_cond_emb, 
                    tgt_cond_emb, 
                    target_text,
                    yaml_file, 
                    controller_type='attn_store',
                    step_recog_loss = 35,
                    NUM_DDIM_STEPS = 50, 
                    num_opt_emb_iter=500, 
                    num_ft_et_iter =1500,
                    update_zT=False,
                    ft_gs=0.3,
                    exp_code = 'original_p2p',
                    scheduler_name='ddim',
                    save_dir='debug/finetune', 
                    device = 'cuda'):
    logger.info(f'Running protocol 9')
    save_sampling_dir = os.path.join(save_dir, 'p9', scheduler_name)
    if controller_type=='attn_store':
        attn = AttentionStore()

    if exp_code =='original_p2p':
        context = torch.cat([uncond_emb, tgt_cond_emb.detach()])
        if scheduler_name== 'ddim':
            z0 = denoising_utils.run_ddim_p_sample_with_gs_p2p(ldm_stable, scheduler, attn, 
                                zT, context, guidance_scale=ft_gs, save_dir=save_sampling_dir)
        else:
            z0 = denoising_utils.run_custom_p_sample_with_gs_p2p(ldm_stable, scheduler, attn, 
                                zT, context, guidance_scale=ft_gs, save_dir=save_sampling_dir)
    
    elif exp_code =='p2p_ft_et_mse':
        run_finetune_et_model(ldm_stable, scheduler, z0, src_cond_emb.detach(), 
                            num_iter=num_ft_et_iter,device=device)

        context = torch.cat([uncond_emb, tgt_cond_emb.detach()])
        if scheduler_name== 'ddim':
            z0 = denoising_utils.run_ddim_p_sample_with_gs_p2p(ldm_stable, scheduler, attn, 
                                zT, context, guidance_scale=ft_gs, save_dir=save_sampling_dir)
        else:
            z0 = denoising_utils.run_custom_p_sample_with_gs_p2p(ldm_stable, scheduler, attn, 
                                zT, context, guidance_scale=ft_gs, save_dir=save_sampling_dir)

    elif exp_code =='p2p_ft_et_mse_opt_tgt_text_recog':
        run_finetune_et_model(ldm_stable, scheduler, z0, src_cond_emb.detach(), 
                            num_iter=num_ft_et_iter,device=device)
        opt_tgt_emb = optimize_emb_with_text_recog_loss(ldm_stable, scheduler, zT, 
                                uncond_emb, tgt_cond_emb, 
                                target_text= target_text,
                                yaml_file= yaml_file, 
                                num_iter = 100 ,
                                guidance_scale=ft_gs,
                                step_recog_loss = step_recog_loss,
                                use_lr_decay=True,
                                lr_type='steplr',
                                save_dir=save_dir)
        context = torch.cat([uncond_emb, opt_tgt_emb])
        if scheduler_name== 'ddim':
            z0 = denoising_utils.run_ddim_p_sample_with_gs_p2p(ldm_stable, scheduler, attn, 
                                zT, context, guidance_scale=ft_gs, save_dir=save_sampling_dir)
        else:
            z0 = denoising_utils.run_custom_p_sample_with_gs_p2p(ldm_stable, scheduler, attn, 
                                zT, context, guidance_scale=ft_gs, save_dir=save_sampling_dir)       

    elif exp_code =='p2p_opt_tgt_text_recog_ft_et_mse':
        opt_tgt_emb = optimize_emb_with_text_recog_loss(ldm_stable, scheduler, zT, 
                                uncond_emb, tgt_cond_emb, 
                                target_text= target_text,
                                yaml_file= yaml_file, 
                                num_iter = 100 ,
                                guidance_scale=ft_gs,
                                step_recog_loss = step_recog_loss,
                                use_lr_decay=True,
                                lr_type='steplr',
                                save_dir=save_dir)

        run_finetune_et_model(ldm_stable, scheduler, z0, src_cond_emb.detach(), 
                            num_iter=num_ft_et_iter,device=device)
        
        context = torch.cat([uncond_emb, opt_tgt_emb])
        if scheduler_name== 'ddim':
            z0 = denoising_utils.run_ddim_p_sample_with_gs_p2p(ldm_stable, scheduler, attn, 
                                zT, context, guidance_scale=ft_gs, save_dir=save_sampling_dir)
        else:
            z0 = denoising_utils.run_custom_p_sample_with_gs_p2p(ldm_stable, scheduler, attn, 
                                zT, context, guidance_scale=ft_gs, save_dir=save_sampling_dir)    

    elif exp_code =='p2p_ft_et_mse_opt_tgt_ft_et_text_recog':
        run_finetune_et_model(ldm_stable, scheduler, z0, src_cond_emb.detach(), 
                            num_iter=num_ft_et_iter,device=device)
        opt_tgt_emb = optimize_emb_with_text_recog_loss(ldm_stable, scheduler, zT, 
                                uncond_emb, tgt_cond_emb, 
                                target_text= target_text,
                                yaml_file= yaml_file, 
                                num_iter = 100 ,
                                guidance_scale=ft_gs,
                                step_recog_loss = step_recog_loss,
                                use_lr_decay=True,
                                lr_type='steplr',
                                save_dir=save_dir)
        run_finetune_et_model_with_text_recog(ldm_stable, scheduler, zT,
                    uncond_emb, opt_tgt_emb.detach(), 
                    target_text= target_text,
                    yaml_file= yaml_file, 
                    num_iter= 100 ,
                    guidance_scale=ft_gs,
                    step_recog_loss = step_recog_loss,
                    save_dir=save_dir)
        context = torch.cat([uncond_emb, opt_tgt_emb])
        if scheduler_name== 'ddim':
            z0 = denoising_utils.run_ddim_p_sample_with_gs_p2p(ldm_stable, scheduler, attn, 
                                zT, context, guidance_scale=ft_gs, save_dir=save_sampling_dir)
        else:
            z0 = denoising_utils.run_custom_p_sample_with_gs_p2p(ldm_stable, scheduler, attn, 
                                zT, context, guidance_scale=ft_gs, save_dir=save_sampling_dir)    

    elif exp_code =='p2p_ft_et_mse_ft_et_text_recog':
        run_finetune_et_model(ldm_stable, scheduler, z0, src_cond_emb.detach(), 
                            num_iter=num_ft_et_iter,device=device)
        run_finetune_et_model_with_text_recog(ldm_stable, scheduler, zT,
                    uncond_emb, tgt_cond_emb.detach(), 
                    target_text= target_text,
                    yaml_file= yaml_file, 
                    num_iter= 100 ,
                    guidance_scale=ft_gs,
                    step_recog_loss = step_recog_loss,
                    use_lr_decay=True,
                    lr_type='steplr',
                    save_dir=save_dir)
        context = torch.cat([uncond_emb, tgt_cond_emb.detach()])
        z0 = denoising_utils.run_ddim_p_sample_with_gs_p2p(ldm_stable, scheduler, attn, 
                            zT, context, guidance_scale=0.3, save_dir=save_sampling_dir)

    x0 =img_utils.latent2im(ldm_stable, z0)
    tvu.save_image((x0 + 1) * 0.5, os.path.join(save_sampling_dir, f'[{exp_code}]_x0_rec.jpg'))
    
    #--Show cross attention
    save_attn_dir = os.path.join(save_sampling_dir, 'attn')
    if not os.path.exists(save_attn_dir):
        os.makedirs(save_attn_dir)
    show_cross_attention(tokenizer, tgt_prp, attn, res=16, from_where=["up", "down"], 
                        save_dir=save_attn_dir, save_name=f'[{exp_code}]')
def run_protocol_10(ldm_stable, tokenizer, scheduler, 
                    z0, zT, src_prp, tgt_prp,
                    uncond_emb, src_cond_emb, 
                    tgt_cond_emb, 
                    target_text,
                    yaml_file, 
                    controller_type='attn_store',
                    step_recog_loss = 35,
                    NUM_DDIM_STEPS = 50, 
                    num_opt_emb_iter=500, 
                    alpha_mix = 0.4, 
                    num_ft_et_iter =1500,
                    update_zT=False,
                    ft_gs=0.3,
                    start_idx_mix=10,
                    using_hybrid =False,
                    exp_code = 'original_p2p',
                    scheduler_name='ddim',
                    save_dir='debug/finetune', 
                    device = 'cuda'):
    logger.info(f'Running protocol 10')
    save_sampling_dir = os.path.join(save_dir, 'p10', scheduler_name)
    
    if controller_type=='attn_store':
        attn = AttentionStore()

    #--Mixture with specific iteration
    if exp_code == 'p2p_ft_et_mse':
        run_finetune_et_model(ldm_stable, scheduler, z0, src_cond_emb.detach(), 
                                num_iter=num_ft_et_iter,device=device)
        context = torch.cat([uncond_emb, src_cond_emb.detach(), tgt_cond_emb.detach()])

    elif exp_code == 'p2p_ft_et_mse_opt_src':
        run_finetune_et_model(ldm_stable, scheduler, z0, src_cond_emb.detach(), 
                                num_iter=num_ft_et_iter,device=device)
        opt_src_emb = run_optimize_emb(ldm_stable, scheduler, z0, src_cond_emb, 
                                    num_iter=num_opt_emb_iter, device=device)
    
        context = torch.cat([uncond_emb, opt_src_emb.detach(), tgt_cond_emb.detach()])

    elif exp_code == 'p2p_opt_src_ft_et_mse':
        opt_src_emb = run_optimize_emb(ldm_stable, scheduler, z0, src_cond_emb, 
                                    num_iter=num_opt_emb_iter, device=device)
        run_finetune_et_model(ldm_stable, scheduler, z0, src_cond_emb.detach(), 
                                num_iter=num_ft_et_iter,device=device)
        context = torch.cat([uncond_emb, opt_src_emb.detach(), tgt_cond_emb.detach()])
    elif exp_code == 'p2p_ft_et_mse_ft_et_text_recog':
        run_finetune_et_model(ldm_stable, scheduler, z0, src_cond_emb.detach(), 
                                num_iter=num_ft_et_iter,device=device)
        run_finetune_et_model_with_text_recog(ldm_stable, scheduler, zT,
                    uncond_emb, tgt_cond_emb.detach(), 
                    target_text= target_text,
                    yaml_file= yaml_file, 
                    num_iter= 100 ,
                    guidance_scale=ft_gs,
                    step_recog_loss = step_recog_loss,
                    save_dir=save_dir)
        context = torch.cat([uncond_emb, src_cond_emb.detach(), tgt_cond_emb.detach()])

    #--Sampling
    if using_hybrid:
        if scheduler_name== 'ddim':
            z0 = denoising_utils.run_ddim_mixture_p_sample_with_norm_gs_p2p_hybrid(ldm_stable, scheduler, attn, 
                                zT, context, guidance_scale=ft_gs, alpha_mix =alpha_mix, start_idx_mix= start_idx_mix,
                                save_dir=save_sampling_dir)
    else:
        if scheduler_name== 'ddim':
            z0 = denoising_utils.run_ddim_mixture_p_sample_with_norm_gs_p2p(ldm_stable, scheduler, attn, 
                                zT, context, guidance_scale=ft_gs, alpha_mix =alpha_mix, 
                                save_dir=save_sampling_dir)
        else:
            z0 = denoising_utils.run_custom_mixture_p_sample_with_norm_gs_p2p(ldm_stable, scheduler, attn, 
                                zT, context, alpha_mix =alpha_mix, guidance_scale=ft_gs, save_dir=save_sampling_dir)
    x0 =img_utils.latent2im(ldm_stable, z0)
    tvu.save_image((x0 + 1) * 0.5, os.path.join(save_sampling_dir, f'[{exp_code}]_x0_rec.jpg'))
    
    #--Show cross attention
    save_attn_dir = os.path.join(save_sampling_dir, 'attn')
    if not os.path.exists(save_attn_dir):
        os.makedirs(save_attn_dir)
    show_cross_attention(tokenizer, tgt_prp, attn, res=16, from_where=["up", "down"], 
                        save_dir=save_attn_dir, save_name=f'[{exp_code}]')

def run_protocol_11(ldm_stable, tokenizer, scheduler, 
                    z0, zT, src_prp, tgt_prp,
                    uncond_emb, src_cond_emb, 
                    tgt_cond_emb, 
                    target_text,
                    yaml_file, 
                    controller_type='attn_store',
                    step_recog_loss = 35,
                    NUM_DDIM_STEPS = 50, 
                    num_opt_emb_iter=500, 
                    alpha_mix = 0.4, 
                    num_ft_et_iter =1500,
                    update_zT=False,
                    ft_gs=0.3,
                    start_idx_mix=10,
                    exp_code = 'original_p2p',
                    scheduler_name='ddim',
                    save_dir='debug/finetune', 
                    device = 'cuda'):
    logger.info(f'Running protocol 11')
    save_sampling_dir = os.path.join(save_dir, 'p11', scheduler_name)
    
    if controller_type=='attn_store':
        attn = AttentionStore()

    #--Mixture with specific iteration
    if exp_code == 'p2p_ft_et_mse':
        logger.debug('Run p2p + finetune et with MSE')
        run_finetune_et_model(ldm_stable, scheduler, z0, src_cond_emb.detach(), 
                                num_iter=num_ft_et_iter,device=device)
        context = torch.cat([uncond_emb, src_cond_emb.detach(), tgt_cond_emb.detach()])

    elif exp_code == 'p2p_ft_et_mse_opt_src':
        run_finetune_et_model(ldm_stable, scheduler, z0, src_cond_emb.detach(), 
                                num_iter=num_ft_et_iter,device=device)
        opt_src_emb = run_optimize_emb(ldm_stable, scheduler, z0, src_cond_emb, 
                                    num_iter=num_opt_emb_iter, device=device)
        context = torch.cat([uncond_emb, opt_src_emb.detach(), tgt_cond_emb.detach()])

    elif exp_code == 'p2p_opt_src_ft_et_mse':
        opt_src_emb = run_optimize_emb(ldm_stable, scheduler, z0, src_cond_emb, 
                                    num_iter=num_opt_emb_iter, device=device)
        run_finetune_et_model(ldm_stable, scheduler, z0, src_cond_emb.detach(), 
                                num_iter=num_ft_et_iter,device=device)
        context = torch.cat([uncond_emb, opt_src_emb.detach(), tgt_cond_emb.detach()])

    elif exp_code == 'p2p_opt_src_ft_et_mse_opt_tgt_text_recog':
        logger.debug('Run p2p + opt src cond + finetune et with MSE + finetune et with text recog')
        opt_src_emb = run_optimize_emb(ldm_stable, scheduler, z0, src_cond_emb, 
                                    num_iter=num_opt_emb_iter, device=device)
        run_finetune_et_model(ldm_stable, scheduler, z0, src_cond_emb.detach(), 
                                num_iter=num_ft_et_iter,device=device)
        run_finetune_et_model_with_text_recog(ldm_stable, scheduler, zT,
                    uncond_emb, tgt_cond_emb.detach(), 
                    target_text= target_text,
                    yaml_file= yaml_file, 
                    num_iter= 100 ,
                    guidance_scale=ft_gs,
                    step_recog_loss = step_recog_loss,
                    use_lr_decay=True,
                    lr_type='steplr',
                    save_dir=save_dir)
        context = torch.cat([uncond_emb, opt_src_emb.detach(), tgt_cond_emb.detach()])

    elif exp_code == 'p2p_opt_src_ft_et_mse_opt_tgt_text_recog':
        logger.debug('Run p2p + opt src cond + finetune et with MSE + opt tgt cond text recog')
        opt_src_emb = run_optimize_emb(ldm_stable, scheduler, z0, src_cond_emb, 
                                    num_iter=num_opt_emb_iter, device=device)
        run_finetune_et_model(ldm_stable, scheduler, z0, src_cond_emb.detach(), 
                                num_iter=num_ft_et_iter,device=device)
        opt_tgt_emb = optimize_emb_with_text_recog_loss(ldm_stable, scheduler, zT, 
                                uncond_emb, tgt_cond_emb, 
                                target_text= target_text,
                                yaml_file= yaml_file, 
                                num_iter = 100 ,
                                guidance_scale=ft_gs,
                                step_recog_loss = step_recog_loss,
                                use_lr_decay=True,
                                lr_type='steplr',
                                save_dir=save_dir)
        context = torch.cat([uncond_emb, opt_src_emb.detach(), opt_tgt_emb.detach()])

    elif exp_code == 'p2p_opt_src_ft_et_mse_ft_et_text_recog':
        logger.debug('Run p2p + opt src cond + finetune et with MSE + finetune et with text recog')
        opt_src_emb = run_optimize_emb(ldm_stable, scheduler, z0, src_cond_emb, 
                                    num_iter=num_opt_emb_iter, device=device)
        run_finetune_et_model(ldm_stable, scheduler, z0, opt_src_emb.detach(), 
                                num_iter=num_ft_et_iter,device=device)
        run_finetune_et_model_with_text_recog(ldm_stable, scheduler, zT,
                    uncond_emb, tgt_cond_emb.detach(), 
                    target_text= target_text,
                    yaml_file= yaml_file, 
                    num_iter= 100 ,
                    guidance_scale=ft_gs,
                    step_recog_loss = step_recog_loss,
                    use_lr_decay=True,
                    lr_type='steplr',
                    save_dir=save_dir)               
        context = torch.cat([uncond_emb, opt_src_emb.detach(), tgt_cond_emb.detach()])
    elif exp_code == 'p2p_ft_et_mse_ft_et_text_recog':
        logger.debug('Run p2p + finetune et with MSE + finetune et with text recog')
        
        run_finetune_et_model(ldm_stable, scheduler, z0, src_cond_emb.detach(), 
                                num_iter=num_ft_et_iter,device=device)
        run_finetune_et_model_with_text_recog(ldm_stable, scheduler, zT,
                    uncond_emb, tgt_cond_emb.detach(), 
                    target_text= target_text,
                    yaml_file= yaml_file, 
                    num_iter= 100 ,
                    guidance_scale=ft_gs,
                    step_recog_loss = step_recog_loss,
                    use_lr_decay=True,
                    lr_type='steplr',
                    save_dir=save_dir)               
        context = torch.cat([uncond_emb, src_cond_emb.detach(), tgt_cond_emb.detach()])
    #--Sampling
   
    if scheduler_name== 'ddim':
        z0 = denoising_utils.run_ddim_mixture_p_sample_with_norm_gs_p2p_hybrid(ldm_stable, scheduler, attn, 
                            zT, context, guidance_scale=ft_gs, alpha_mix =alpha_mix, start_idx_mix= start_idx_mix,
                            save_dir=save_sampling_dir)
    else:
        z0 = denoising_utils.run_custom_mixture_p_sample_with_norm_gs_p2p_hybrid(ldm_stable, scheduler, attn, 
                            zT, context, guidance_scale=ft_gs, alpha_mix =alpha_mix, start_idx_mix= start_idx_mix,
                            save_dir=save_sampling_dir)
    x0 =img_utils.latent2im(ldm_stable, z0)
    tvu.save_image((x0 + 1) * 0.5, os.path.join(save_sampling_dir, f'[{exp_code}]_x0_rec.jpg'))
    
    #--Show cross attention
    save_attn_dir = os.path.join(save_sampling_dir, 'attn')
    if not os.path.exists(save_attn_dir):
        os.makedirs(save_attn_dir)
    show_cross_attention(tokenizer, tgt_prp, attn, res=16, from_where=["up", "down"], 
                        save_dir=save_attn_dir, save_name=f'[{exp_code}]')

def run_simon_idea(ldm_stable, tokenizer, scheduler, 
                    z0, zT, src_prp, tgt_prp,
                    uncond_emb, src_cond_emb, 
                    tgt_cond_emb, 
                    target_text,
                    yaml_file, 
                    controller_type='attn_store',
                    step_recog_loss = 35,
                    NUM_DDIM_STEPS = 50, 
                    num_opt_emb_iter=500, 
                    alpha_mix = 0.4, 
                    num_ft_et_iter =1500,
                    update_zT=False,
                    ft_gs=0.3,
                    start_idx_mix=10,
                    exp_code = 'original_p2p',
                    scheduler_name='ddim',
                    save_dir='debug/finetune', 
                    pretrained_et_aug=None,
                    img = None,
                    device = 'cuda'):
    logger.info(f'Running protocol simon idea')
    save_sampling_dir = os.path.join(save_dir, 'simon_idea', scheduler_name)
    
    if controller_type=='attn_store':
        attn = AttentionStore()
    # context = torch.cat([uncond_emb, src_cond_emb.detach(), tgt_cond_emb.detach()])
    # context = torch.cat([uncond_emb, src_cond_emb.detach()])
    # context = torch.cat([uncond_emb, tgt_cond_emb.detach()])
    #--Load finetuned
    # if pretrained_et_aug != None:
    #     et_weight_augm = torch.load(pretrained_et_aug, map_location=torch.device('cpu'))
    #     ldm_stable.unet.load_state_dict(et_weight_augm)
    # else:
    #     run_finetune_et_model(ldm_stable, scheduler, z0, src_cond_emb.detach(), 
    if exp_code =='opt_src_augm_mse':
        #--Optimize source embedding using augmentation
        opt_src_augm_emb = run_optimize_emb_augm(ldm_stable, scheduler, img, src_cond_emb, 
                                    num_iter=5000, device=device)
        context = torch.cat([uncond_emb, opt_src_augm_emb.detach()])
    if exp_code =='opt_src_mse':
        #--Optimize source embedding using augmentation
        opt_src_emb = run_optimize_emb(ldm_stable, scheduler, z0, src_cond_emb, 
                                    num_iter=7500, device=device)
        context = torch.cat([uncond_emb, opt_src_emb.detach()])
    if exp_code =='opt_src_mse_text_recog':
        #--Optimize source embedding using augmentation
        opt_src_emb = run_optimize_emb(ldm_stable, scheduler, z0, src_cond_emb, 
                                    num_iter=7500, device=device)
        opt_src_emb = optimize_emb_with_text_recog_loss(ldm_stable, scheduler, zT, 
                                uncond_emb, opt_src_emb.detach(), 
                                target_text= target_text,
                                yaml_file= yaml_file, 
                                num_iter = 1000 ,
                                lr=0.0001,
                                guidance_scale=ft_gs,
                                step_recog_loss = step_recog_loss,
                                use_lr_decay=True,
                                lr_type='steplr',
                                save_dir=save_dir)
        context = torch.cat([uncond_emb, opt_src_emb.detach()])
    
        context = torch.cat([uncond_emb, opt_tgt_emb.detach()])
    elif exp_code =='opt_tgt_text_recog':
        opt_tgt_emb = optimize_emb_with_text_recog_loss(ldm_stable, scheduler, zT, 
                                uncond_emb, tgt_cond_emb, 
                                target_text= target_text,
                                yaml_file= yaml_file, 
                                num_iter = 100 ,
                                guidance_scale=ft_gs,
                                step_recog_loss = step_recog_loss,
                                use_lr_decay=True,
                                lr_type='steplr',
                                save_dir=save_dir)
        context = torch.cat([uncond_emb, opt_tgt_emb.detach()])
    elif exp_code =='opt_src_mse':
        opt_src_emb = run_optimize_emb(ldm_stable, scheduler, z0, src_cond_emb, 
                                    num_iter=num_opt_emb_iter, device=device)
        context = torch.cat([uncond_emb, opt_src_emb.detach()])
    #--Sampling
    z0 = denoising_utils.run_ddim_p_sample_with_gs_p2p(ldm_stable, scheduler, attn, 
                            zT, context, guidance_scale=ft_gs, save_dir=save_sampling_dir)
    # if scheduler_name== 'ddim':
    #     z0 = denoising_utils.run_ddim_mixture_p_sample_with_norm_gs_p2p_hybrid(ldm_stable, scheduler, attn, 
    #                         zT, context, guidance_scale=ft_gs, alpha_mix =alpha_mix, start_idx_mix= start_idx_mix,
    #                         save_dir=save_sampling_dir)
    # else:
    #     z0 = denoising_utils.run_custom_mixture_p_sample_with_norm_gs_p2p_hybrid(ldm_stable, scheduler, attn, 
    #                         zT, context, guidance_scale=ft_gs, alpha_mix =alpha_mix, start_idx_mix= start_idx_mix,
    #                         save_dir=save_sampling_dir)
    x0 =img_utils.latent2im(ldm_stable, z0)
    tvu.save_image((x0 + 1) * 0.5, os.path.join(save_sampling_dir, f'[{exp_code}]_x0_rec.jpg'))
    
    #--Show cross attention
    save_attn_dir = os.path.join(save_sampling_dir, 'attn')
    if not os.path.exists(save_attn_dir):
        os.makedirs(save_attn_dir)
    show_cross_attention(tokenizer, src_prp, attn, res=16, from_where=["up", "down"], 
                        save_dir=save_attn_dir, save_name=f'[{exp_code}]')
    # show_cross_attention(tokenizer, tgt_prp, attn, res=16, from_where=["up", "down"], 
    #                     save_dir=save_attn_dir, save_name=f'[{exp_code}]')
    