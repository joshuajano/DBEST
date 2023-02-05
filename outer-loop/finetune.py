import os 
import torch
import torchvision.utils as tvu
import numpy as np 
from PIL import Image
# from model_512 import ldm_stable, tokenizer, scheduler, device, MAX_NUM_WORDS, NUM_DDIM_STEPS
from model_256 import ldm, tokenizer, scheduler, device, MAX_NUM_WORDS, NUM_DDIM_STEPS
from utils import img_utils, diffusion_utils, prompt_utils, denoising_utils
from baselines.imagic.method import run_optimize_emb, run_finetune_et_model, run_finetune_et_model_with_syntext, \
                                    run_finetune_et_model_with_syntext_from_path, run_finetune_et_model_input_image
from methods.optimization import optimize_emb_with_text_recog_loss
from methods.experiments import run_protocol_1, run_protocol_2, run_protocol_3, run_protocol_4, run_protocol_5, run_protocol_6
g = torch.Generator(device=device).manual_seed(888)
try:
    ldm_stable = ldm
except:
    ldm_stable = ldm_stable

#--Train from images_path
dataset_dir = '/media/verihubs/verihubs-2TB/datasets/text-scene/syntext345chars_500k/'
texts_list = torch.load(os.path.join(dataset_dir, 'texts.ckpt'), map_location='cpu')
imgs_dir = os.path.join(dataset_dir, 't_f')
run_finetune_et_model_with_syntext_from_path(ldm_stable, tokenizer, scheduler, texts_list, 
                num_iter=1000000, dataset_dir=imgs_dir, device=device)

#--Finetune inner model with simon idea
# target_size = 256
# save_dir = 'debug_images/'
# imgs_dir = '/home/verihubs/Documents/joshua/stable-diffusion/im-examples/'
# img_name = 'ic13/142.png'

# et_weight_syntext = torch.load('data/ft_et_text_syntext4chars_100k.ckpt', map_location=torch.device('cpu'))

# #--Update the weight
# ldm_stable.unet.load_state_dict(et_weight_syntext)

# save_path = os.path.join(save_dir, img_name[:-4], str(target_size))
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# img_path = os.path.join(imgs_dir, img_name)
# src_img = img_utils.load_np_image(img_path, target_size=target_size)

# src_prompt = ["A text that reads 'JUST' "]
# with torch.no_grad():
#     uncond_emb, src_cond_emb = prompt_utils.gen_init_prompt_to_emb(ldm_stable, tokenizer, 
#                                 src_prompt, MAX_NUM_WORDS, device=device)
# run_finetune_et_model_input_image(ldm_stable, scheduler, src_img, src_cond_emb.detach(),
#                 num_iter=5000,device=device)
# augmented_img = img_utils.augment_and_norm_image(src_img).to(device).unsqueeze(0)
# tvu.save_image((augmented_img + 1) * 0.5, 'test.jpg')
# prep_data_dir = 'data/'
# prep_data_dir_100k = '/media/verihubs/verihubs-2TB/datasets/text-scene/prep_tensor_syntext/'




# data_1 = torch.load(os.path.join(prep_data_dir_100k, 'prep_100k_words_syntext4chars_10000.ckpt'), map_location='cpu')
# data_2 = torch.load(os.path.join(prep_data_dir_100k, 'prep_100k_words_syntext4chars_20000.ckpt'), map_location='cpu')
# data_3 = torch.load(os.path.join(prep_data_dir_100k, 'prep_100k_words_syntext4chars_20000.ckpt'), map_location='cpu')
# data_1.update(data_2)
# prep_data = data_1



# prep_data = torch.load(os.path.join(prep_data_dir, 'prep_words.ckpt'))
# prep_data = torch.load(os.path.join(prep_data_dir, 'prep_3000_words_syntext4chars.ckpt'))

# run_finetune_et_model_with_syntext(ldm_stable, scheduler, prep_data, 
#                 num_iter=200000,device=device)

# run_finetune_et_model_with_syntext(ldm_stable, scheduler, prep_data, 
#                 num_iter=200000,device=device)

