import os 
import torch
import torchvision.utils as tvu
import numpy as np 
from PIL import Image
import cv2

from loguru import logger

from model_256 import ldm, tokenizer, scheduler, PNDM_scheduler, device, MAX_NUM_WORDS, NUM_DDIM_STEPS
from utils import img_utils, diffusion_utils, prompt_utils, denoising_utils

from baselines.imagic.method import run_optimize_emb, run_finetune_et_model, \
            run_finetune_et_model_cycle_loss, run_finetune_et_model_idn_loss

from methods.optimization import optimize_emb_with_text_recog_loss, opt_emb_DDPM_text_recog, run_finetune_et_model_text_recog, \
                                run_finetune_et_model_text_recog_style, opt_emb_DDPM_text_recog_style
from exp_preserve_style import run_p2p_cycle

from baselines.prompt2prompt.method import AttentionStore, show_cross_attention
def get_mean_and_std(x):
	x_mean, x_std = cv2.meanStdDev(x)
	x_mean = np.hstack(np.around(x_mean,2))
	x_std = np.hstack(np.around(x_std,2))
	return x_mean, x_std
#--All configs should be put on configs.py
import configs
g = torch.Generator(device=device).manual_seed(888)
try:
    ldm_stable = ldm
except:
    ldm_stable = ldm_stable
logger.warning(f'Load pretrained syntext {configs.PRETRAINED_WEIGHT_PATH_SYNTEXT}')
et_weight_syntext = torch.load(configs.PRETRAINED_WEIGHT_PATH_SYNTEXT, map_location=torch.device('cpu'))
#--Update the weight
ldm_stable.unet.load_state_dict(et_weight_syntext)

save_path = os.path.join(configs.SAVE_DIR, configs.DSET_NAME, configs.IMG_NAME[:-4])
if not os.path.exists(save_path):
    os.makedirs(save_path)
scheduler_name = 'ddim'
controller = AttentionStore()    

#--Get source image
src_img_path = os.path.join(configs.IMGS_DIR, configs.DSET_NAME, configs.IMG_NAME)
src_uncond_emb, src_cond_emb = prompt_utils.gen_init_prompt_to_emb(ldm_stable, tokenizer, 
                            configs.SOURCE_PROMPT, MAX_NUM_WORDS, device=device)
src_context = torch.cat([src_uncond_emb, src_cond_emb])
src_z0, src_zT = img_utils.prepare_input(ldm_stable, scheduler, 
                        src_context, src_img_path, g, 
                        NUM_DDIM_STEPS, configs.IMG_SIZE, 
                        save_path, device)
# if os.path.exists(f'data/ft_et_{configs.DSET_NAME}_{configs.IMG_NAME[:-4]}.ckpt'):
#     logger.info("pretrained found")
#     word_et_weight = torch.load(f'data/ft_et_{configs.DSET_NAME}_{configs.IMG_NAME[:-4]}.ckpt', map_location=torch.device('cpu'))
#     #--Update the weight
#     ldm_stable.unet.load_state_dict(et_weight_syntext)
# else:
# logger.info("pretrained not found, start finetune" )
# yaml_file = '/home/verihubs/Documents/joshua/ABINet/configs/train_abinet.yaml'
# tgt_uncond_emb, tgt_cond_emb = prompt_utils.gen_init_prompt_to_emb(ldm_stable, tokenizer, 
#                                 configs.TARGET_PROMPT, MAX_NUM_WORDS, device=device)

# tgt_context = torch.cat([tgt_uncond_emb, tgt_cond_emb])
# run_finetune_et_model_text_recog_style(ldm_stable, ldm_stable.scheduler, src_z0, 
#                                             tgt_cond_emb.detach(), 
#                                             target_text= configs.TARGET_TEXT,
#                                             yaml_file= yaml_file, 
#                                             num_iter = 1000)
run_finetune_et_model(ldm_stable, ldm_stable.scheduler, src_z0.detach(), src_cond_emb.detach(), 
                    num_iter=configs.FT_ET_ITER, device=device)
# torch.save(ldm_stable.unet.state_dict(), f'data/ft_et_{configs.DSET_NAME}_{configs.IMG_NAME[:-4]}.ckpt')
tgt_uncond_emb, tgt_cond_emb = prompt_utils.gen_init_prompt_to_emb(ldm_stable, tokenizer, 
                                configs.TARGET_PROMPT, MAX_NUM_WORDS, device=device)

tgt_context = torch.cat([tgt_uncond_emb, tgt_cond_emb])
save_sampling_dir = os.path.join(save_path, 'image_level', scheduler_name)
#--Sampling
do_mixture = True
do_rand_noise = False
do_opt_emb = True
if do_mixture:
    # opt_tgt_emb =
    yaml_file = '/home/verihubs/Documents/joshua/ABINet/configs/train_abinet.yaml'
    save_sampling_dir = os.path.join(save_sampling_dir, 'mixture')
    if do_opt_emb:
        # opt_tgt_emb = opt_emb_DDPM_text_recog(ldm_stable, ldm_stable.scheduler, src_z0, 
        #                                     src_cond_emb,
        #                                     tgt_cond_emb, 
        #                                     target_text= configs.TARGET_TEXT,
        #                                     yaml_file= yaml_file, 
        #                                     num_iter = 1000,
        #                                     save_dir=save_sampling_dir)
        opt_tgt_emb = opt_emb_DDPM_text_recog_style(ldm_stable, ldm_stable.scheduler, src_z0, 
                                            src_cond_emb,
                                            tgt_cond_emb, 
                                            target_text= configs.TARGET_TEXT,
                                            yaml_file= yaml_file, 
                                            num_iter = 1000,
                                            save_dir=save_sampling_dir)

        # run_finetune_et_model_text_recog_style(ldm_stable, ldm_stable.scheduler, src_z0, 
        #                                     opt_tgt_emb.detach(), 
        #                                     target_text= configs.TARGET_TEXT,
        #                                     yaml_file= yaml_file, 
        #                                     num_iter = 1500)
        # run_finetune_et_model_text_recog(ldm_stable, ldm_stable.scheduler, src_z0, 
        #                                     opt_tgt_emb.detach(), 
        #                                     target_text= configs.TARGET_TEXT,
        #                                     yaml_file= yaml_file, 
        #                                     num_iter = 1000)
        #--Ini works
        # opt_tgt_emb = opt_emb_DDPM_text_recog(ldm_stable, ldm_stable.scheduler, src_z0, 

        #                             tgt_cond_emb, 
        #                             target_text= configs.TARGET_TEXT,
        #                             yaml_file= yaml_file, 
        #                             num_iter = 5000 ,
        #                             save_dir=save_sampling_dir)
        context = torch.cat([tgt_uncond_emb, opt_tgt_emb, opt_tgt_emb])
    else:
        context = torch.cat([tgt_uncond_emb, tgt_cond_emb, tgt_cond_emb])
    
    tgt_z0 = denoising_utils.run_ddim_mixture_p_sample_with_norm_gs_p2p(ldm_stable, 
                    ldm_stable.scheduler, 
                    controller, src_zT, context,
                    NUM_DDIM_STEPS = 50,
                    num_inference_steps=50, 
                    start_time = 50, 
                    guidance_scale=configs.GUIDANCE_SCALE,
                    alpha_mix =0.9, 
                    save_dir= save_sampling_dir)
else:
    if do_rand_noise:
        # torch.manual_seed(888)
        src_zT = torch.randn_like(src_z0)
        tvu.save_image((src_zT + 1) * 0.5, os.path.join(save_sampling_dir, f'rand_noise.png'))
    tgt_z0 = denoising_utils.run_ddim_p_sample_with_gs_p2p(ldm_stable, ldm_stable.scheduler, controller, 
                            src_zT, tgt_context, guidance_scale=configs.GUIDANCE_SCALE, save_dir=save_sampling_dir)
src_x0 =img_utils.latent2im(ldm_stable, src_z0)
tvu.save_image((src_x0 + 1) * 0.5, os.path.join(save_sampling_dir, f'x0.png'))
tgt_x0 =img_utils.latent2im(ldm_stable, tgt_z0)
tvu.save_image((tgt_x0 + 1) * 0.5, os.path.join(save_sampling_dir, f'x0_rec.png'))

#--Post process
#--Color transfer 
s_img = cv2.imread(os.path.join(save_sampling_dir, f'x0_rec.png'))
s = cv2.cvtColor(s_img, cv2.COLOR_BGR2LAB)
t_img = cv2.imread(os.path.join(save_sampling_dir, f'x0.png'))
t = cv2.cvtColor(t_img, cv2.COLOR_BGR2LAB)
s_mean, s_std = get_mean_and_std(s)
t_mean, t_std = get_mean_and_std(t)
height, width, channel = s.shape
for i in range(0, height):
	for j in range(0,width):
		for k in range(0, channel):
			x = s[i,j,k]
			x = ((x-s_mean[k])*(t_std[k]/s_std[k]))+t_mean[k]
			# round or +0.5
			x = round(x)
			# boundary check
			x = 0 if x<0 else x
			x = 255 if x>255 else x
			s[i,j,k] = x
s = cv2.cvtColor(s,cv2.COLOR_LAB2BGR)
cv2.imwrite(os.path.join(save_sampling_dir, f'x0_ct.png'), s)

eval_dirs = '/home/verihubs/Documents/joshua/ours/diffText/eval/ours_eval'
cv2.imwrite(os.path.join(eval_dirs, configs.IMG_NAME), s)
