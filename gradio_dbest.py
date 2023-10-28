import os 
import torchvision.transforms as T
import einops
import gradio as gr
from PIL import Image

import cv2
from loguru import logger
import torch
import numpy as np

from t2tldm.model_256 import ldm, tokenizer, scheduler, PNDM_scheduler, device, MAX_NUM_WORDS, NUM_DDIM_STEPS
from t2tldm.utils import img_utils, diffusion_utils, prompt_utils, denoising_utils
from t2tldm.baselines.imagic.method import run_finetune_et_model 
from t2tldm.methods.optimization import opt_emb_DDPM_text_recog 
# define a transform to convert a tensor to PIL image
transform = T.ToPILImage()
#--Function modulate
def modulate_text(input_image, src_prompt, 
                tgt_prompt, 
                guide_scales = [0.2, 0.3, 0.4, 0.5],
                a_prompt = 'A text that reads: ',
                target_size=256):
    # img = cv2.resize(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), (256, 256))
    img = Image.fromarray(np.uint8(input_image))
    img = img.convert("RGB")
    img = img.resize((target_size, target_size))
    list_imgs = []
    logger.success(a_prompt + f'"{src_prompt}"')
    logger.success(a_prompt + f'"{tgt_prompt}"')
    
    g = torch.Generator(device=device).manual_seed(888)
    pretrained_wegiht = 'ft_et_text_syntext4chars_100k.ckpt'
    et_weight_syntext = torch.load(pretrained_wegiht, map_location=torch.device('cpu'))
    ldm.unet.load_state_dict(et_weight_syntext)
    
    src_p = [a_prompt + f'"{src_prompt}"']
    tgt_p = [a_prompt + f'"{tgt_prompt}"']
    src_uncond_emb, src_cond_emb = prompt_utils.gen_init_prompt_to_emb(ldm, tokenizer, 
                            src_p, MAX_NUM_WORDS, device=device)
    src_context = torch.cat([src_uncond_emb, src_cond_emb])
    src_z0, src_zT = img_utils.prepare_input_gradio(ldm, scheduler, 
                          src_context, img, g, 
                          NUM_DDIM_STEPS, 256, device=device)
    
    run_finetune_et_model(ldm, ldm.scheduler, src_z0.detach(), src_cond_emb.detach(), 
                    num_iter=1500, device=device)
    tgt_uncond_emb, tgt_cond_emb = prompt_utils.gen_init_prompt_to_emb(ldm, tokenizer, 
                                tgt_p, MAX_NUM_WORDS, device=device)
    tgt_context = torch.cat([tgt_uncond_emb, tgt_cond_emb])
    yaml_file = 'data/train_abinet.yaml'
    checkpoint_tr = 'weights/best-train-abinet.pth'
    opt_tgt_emb = opt_emb_DDPM_text_recog(ldm, ldm.scheduler, 
                                            src_z0, 
                                            src_cond_emb,
                                            tgt_cond_emb, 
                                            target_text= tgt_prompt.lower(),
                                            yaml_file= yaml_file, 
                                            checkpoints_path = checkpoint_tr,
                                            num_iter = 1000,
                                            save_dir='', device=device)
    list_gs = [0.2, 0.3, 0.4, 0.5]
    # list_gs = [0.2]
    src_x0 =img_utils.latent2im(ldm, src_z0)
    src_x0 = (einops.rearrange(src_x0.detach(), 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    src_x0 = src_x0[0]
    for gs in list_gs:
        context = torch.cat([tgt_uncond_emb, opt_tgt_emb.detach()])
        ddim_tgt_z0 = denoising_utils.run_ddim_p_sample_norm_gs_paper(
                    ldm, src_zT.clone(), context,
                    NUM_DDIM_STEPS = 50,
                    num_inference_steps=50, 
                    start_time = 50, 
                    guidance_scale=gs)
        ddim_tgt_x0 =img_utils.latent2im(ldm, ddim_tgt_z0)
        ddim_tgt_x0 = (einops.rearrange(ddim_tgt_x0.detach(), 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        #--Do color transfer 
        ct_ddim = img_utils.color_transfer(ddim_tgt_x0[0], src_x0)
        # ct_ddim = cv2.resize(ct_ddim, (100, 100))
        list_imgs.append(ct_ddim)
        pndm_tgt_z0 = denoising_utils.run_pndm_p_sample_norm_gs_paper(
                        ldm, src_zT.clone(), context, PNDM_scheduler, 
                        NUM_DDIM_STEPS = 50,
                        num_inference_steps=50, 
                        start_time = 50, 
                        guidance_scale=gs)
        pndm_tgt_x0 =img_utils.latent2im(ldm, pndm_tgt_z0)
        pndm_tgt_x0 = (einops.rearrange(pndm_tgt_x0.detach(), 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        ct_pndm = img_utils.color_transfer(pndm_tgt_x0[0], src_x0)
        # ct_pndm = cv2.resize(ct_pndm, (100, 100))
        list_imgs.append(ct_pndm)
    # for sampling in ['ddim', 'pndm']:
    #     for gs in guide_scales:
    #         list_imgs.append(img)

    return list_imgs


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## On Modulating Text In The Wild with Diffusion Model" )
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(source='upload', type="numpy")
            src_prompt = gr.Textbox(label="Source")
            tgt_prompt = gr.Textbox(label="Target")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=1., value=0.3, step=0.01)
        # with gr.Column():
    with gr.Row():
        with gr.Column(scale=1):
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=4, height='auto')
        # with gr.Column():5
        #     input_image = gr.Image(source='upload', type="numpy")
        #     prompt = gr.Textbox(label="Prompt")5
        #     run_button = gr.Button(label="Run")
    ips = [input_image, src_prompt, tgt_prompt]
    run_button.click(fn=modulate_text, inputs=ips, outputs=[result_gallery])
block.launch(server_name='0.0.0.0')