import torch
from tqdm import tqdm
from loguru import logger
import random
import os 
from fastai.vision import *
import torch
import torchvision.utils as tvu
import math

from t2tldm.losses.text_recog_loss import MultiLosses
from t2tldm.utils import diffusion_utils, img_utils, prompt_utils
from t2tldm.methods.optimization import run_init_text_recog, onehot, run_recognition_model, postprocess
from t2tldm.utils.text_recog_utils import Config, CharsetMapper
# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
def run_optimize_emb_augm(model, scheduler, src_img, orig_emb, 
    num_iter=500, lr=0.001, device="cuda:0"): #num_iter=500
    logger.debug('Run Imagic optimize embedding')
    alphas_cumprod = scheduler.alphas_cumprod
    with torch.no_grad():
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)

    emb = orig_emb.detach().clone()
    emb.requires_grad = True
    # opt = torch.optim.Adam([emb], lr=lr)
    opt = torch.optim.AdamW([emb], lr=lr)
    criteria = torch.nn.MSELoss()
    pbar = tqdm(range(num_iter))
    
    for i in pbar:
        opt.zero_grad()
        z0 = prepare_input_from_input_img(model, src_img.copy())
        noise = torch.randn_like(z0)
        t_enc = torch.randint(1000, (1,), device=device)
        # zt = model.q_sample(z0, t_enc, noise=noise)
        zt = diffusion_utils.run_ddpm_q_sample(z0.clone(), t_enc, noise, 
                sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        et = model.unet(zt, t_enc, encoder_hidden_states=emb)["sample"]
        loss = criteria(et, noise)
        loss.backward()
        pbar.set_postfix({"loss": loss.item()})   
        opt.step()
    return emb
def run_optimize_emb(model, scheduler, z0, orig_emb, 
    num_iter=500, lr=0.001, device="cuda:0"): #num_iter=500
    logger.debug('Run Imagic optimize embedding')
    alphas_cumprod = scheduler.alphas_cumprod
    with torch.no_grad():
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)

    emb = orig_emb.detach().clone()
    emb.requires_grad = True
    # opt = torch.optim.Adam([emb], lr=lr)
    opt = torch.optim.AdamW([emb], lr=lr)
    criteria = torch.nn.MSELoss()
    pbar = tqdm(range(num_iter))
    
    for i in pbar:
        opt.zero_grad()
        noise = torch.randn_like(z0)
        t_enc = torch.randint(1000, (1,), device=device)
        # zt = model.q_sample(z0, t_enc, noise=noise)
        zt = diffusion_utils.run_ddpm_q_sample(z0.clone(), t_enc, noise, 
                sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        et = model.unet(zt, t_enc, encoder_hidden_states=emb)["sample"]
        loss = criteria(et, noise)
        loss.backward()
        pbar.set_postfix({"loss": loss.item()})   
        opt.step()
    return emb

def run_finetune_et_model_with_text_recog(
                    ldm_stable, scheduler, zT, uncond_emb,
                    tgt_cond_emb, target_text,
                    yaml_file = '/home/verihubs/Documents/joshua/ABINet/configs/train_abinet.yaml',
                    checkpoints_path ='/home/verihubs/Documents/joshua/ABINet/workdir/workdir/train-abinet/best-train-abinet.pth', 
                    num_iter = 100,
                    NUM_DDIM_STEPS =50,
                    start_time = 50,  
                    guidance_scale = 5.,
                    alpha_mix = 0.5,
                    lr=1e-6, step_recog_loss = 25, #lr=0.0001,
                    early_stop_eps =1e-5,
                    use_lr_decay=False,
                    lr_type='steplr',
                    save_dir = '',
                    device='cuda'):
    save_path = os.path.join(save_dir, 'generative')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    config = Config(yaml_file)
    txtrec_model, charset = run_init_text_recog(config, checkpoints_path, device)
    uncond_emb = uncond_emb.detach()
    tgt_emb = tgt_cond_emb.detach().clone() 
    ldm_stable.unet.train()
    opt = torch.optim.AdamW(ldm_stable.unet.parameters(), lr=lr)
    # opt = torch.optim.AdamW([emb], lr=lr)
    if use_lr_decay:
        if lr_type =='steplr':
            lr_scheduler =torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.95)
        elif lr_type =='lambdalr':
            lambda1 = lambda epoch: 0.95 ** epoch
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda1)
        else:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', 
            factor=0.1, patience=10, threshold=0.0001, threshold_mode='abs', 
            cooldown=0, min_lr=0, eps=1e-08, verbose=False)
    criteria = MultiLosses(one_hot=True)
    length = tensor(len(target_text) + 1).to(dtype=torch.long).unsqueeze(0) 
    label = charset.get_labels(target_text, case_sensitive=False)
    label = tensor(label).to(dtype=torch.long)
    onehot_label = onehot(label, charset.num_classes).unsqueeze(0).to(device)
    pbar = tqdm(range(num_iter))
    for i in pbar:
        zt = zT.clone()
        context = torch.cat([uncond_emb, tgt_emb])
        ldm_stable.scheduler.set_timesteps(NUM_DDIM_STEPS)
        with tqdm(total=NUM_DDIM_STEPS, desc=f"DDIM generative process {i}") as progress_bar:
            for i, t in enumerate(ldm_stable.scheduler.timesteps[-start_time:]):
                # zt_cat = torch.cat([zt] * 3)
                zt_cat = torch.cat([zt] * 2)
                if i < step_recog_loss:
                    with torch.no_grad():
                        noise_pred = ldm_stable.unet(zt_cat, t, encoder_hidden_states=context.clone())["sample"]
                        # et_uncond, et_src_cond, et_tgt_cond, = noise_pred.chunk(3)
                        et_uncond, et_tgt_cond, = noise_pred.chunk(2)
                        et = et_uncond + \
                        (alpha_mix) * guidance_scale  * ((et_tgt_cond - et_uncond)/ 
                                    torch.norm(et_tgt_cond - et_uncond) * torch.norm(et_uncond))

                        zt = ldm_stable.scheduler.step(et, t, zt)["prev_sample"]
                else:
                    with torch.enable_grad():
                        noise_pred = ldm_stable.unet(zt_cat, t, encoder_hidden_states=context.clone())["sample"]
                        # et_uncond, et_src_cond, et_tgt_cond, = noise_pred.chunk(3)
                        et_uncond, et_tgt_cond, = noise_pred.chunk(2)
                        et = et_uncond + \
                        (alpha_mix) * guidance_scale  * ((et_tgt_cond - et_uncond)/ 
                                    torch.norm(et_tgt_cond - et_uncond) * torch.norm(et_uncond))
                        zt = ldm_stable.scheduler.step(et, t, zt)["prev_sample"]
                    
                    res =run_recognition_model(ldm_stable, txtrec_model, zt)
                    pt_text, _, __ = postprocess(res, charset, config.model_eval)
                    
                    opt.zero_grad(set_to_none=True)
                    loss = criteria(res, onehot_label, length)
                    if pt_text[0] == target_text.lower() and loss>2.:
                        if use_lr_decay:
                            if lr_type =='rop':
                                lr_scheduler.step(loss)
                            else:
                                lr_scheduler.step()
                    if pt_text[0] == target_text.lower() and loss<=1.:
                        break
                    
                    loss.backward()
                    opt.step()
                    zt = zt.detach().clone()
                    progress_bar.set_postfix({"loss": loss.item(), "text": pt_text})   
                tvu.save_image((zt + 1) * 0.5, os.path.join(save_path, f'z{i}.png'))
                progress_bar.update(1)
        if pt_text[0] == target_text.lower() and loss<=1.:
            break
    x0 =img_utils.latent2im(ldm_stable, zt)
    tvu.save_image((x0 + 1) * 0.5, os.path.join(save_path, f'x0_rec_mixture_opt_source_emb.png'))
    ldm_stable.unet.eval()
# def run_finetune_et_model(model, scheduler, img_path, emb, 
#     num_iter=1000, lr=1e-6, device="cuda:0"):
#     logger.debug('Run Imagic finetune et model')
#     alphas_cumprod = scheduler.alphas_cumprod
#     with torch.no_grad():
#         sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
#         sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)
#     model.unet.train()
#     opt = torch.optim.Adam(model.unet.parameters(), lr=lr)
#     criteria = torch.nn.MSELoss()
#     pbar = tqdm(range(num_iter))
#     for i in pbar:
#         with torch.no_grad():
#         opt.zero_grad()
#         noise = torch.randn_like(z0)
#         t_enc = torch.randint(1000, (1,), device=device)
#         zt = diffusion_utils.run_ddpm_q_sample(z0.clone(), t_enc, noise, 
#                 sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        
#         et = model.unet(zt, t_enc, encoder_hidden_states=emb)["sample"]
#         loss = criteria(et, noise)
#         loss.backward()
#         pbar.set_postfix({"loss": loss.item()})   
#         opt.step()
#     model.unet.eval()
def run_finetune_et_model_cycle(model, scheduler, src_z0, tgt_z0, src_emb, tgt_emb, 
    num_iter=1000, lr=1e-6, device="cuda:0"):
    logger.debug('Run Imagic finetune et cycle model')
    alphas_cumprod = scheduler.alphas_cumprod
    with torch.no_grad():
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)
    model.unet.train()
    opt = torch.optim.Adam(model.unet.parameters(), lr=lr)
    criteria = torch.nn.MSELoss()
    pbar = tqdm(range(num_iter))
    for i in pbar:
        opt.zero_grad()
        noise = torch.randn_like(z0)
        t_enc = torch.randint(1000, (1,), device=device)
        zt = diffusion_utils.run_ddpm_q_sample(z0.clone(), t_enc, noise, 
                sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        
        et = model.unet(zt, t_enc, encoder_hidden_states=emb)["sample"]
        loss = criteria(et, noise)
        loss.backward()
        pbar.set_postfix({"loss": loss.item()})   
        opt.step()
    model.unet.eval()
def run_finetune_et_model(model, scheduler, z0, emb, 
    num_iter=1000, lr=1e-6, device="cuda:0"):
    logger.debug('Run Imagic finetune et model')
    alphas_cumprod = scheduler.alphas_cumprod
    with torch.no_grad():
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)
    model.unet.train()
    # opt = torch.optim.Adam(model.unet.parameters(), lr=lr)
    opt = torch.optim.AdamW(model.unet.parameters(), lr=lr)
    criteria = torch.nn.MSELoss()
    pbar = tqdm(range(num_iter))
    for i in pbar:
        opt.zero_grad()
        noise = torch.randn_like(z0)
        t_enc = torch.randint(1000, (1,), device=device)
        zt = diffusion_utils.run_ddpm_q_sample(z0.clone(), t_enc, noise, 
                sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        
        et = model.unet(zt, t_enc, encoder_hidden_states=emb)["sample"]
        loss = criteria(et, noise)
        loss.backward()
        pbar.set_postfix({"loss": loss.item()})   
        opt.step()
    model.unet.eval()
def get_mean_std_img(img):
    #--Convert range to 0 - 1
    dnorm_img = (img + 1) * 0.5
    b_total_pixel = dnorm_img.shape[0] * dnorm_img.shape[2] * dnorm_img.shape[3]
    pixel_sum = torch.sum(dnorm_img, dim=(0, 2, 3))
    total_mean = pixel_sum / b_total_pixel
    psum_sq =  torch.sum(torch.pow(dnorm_img, 2) , dim=(0, 2, 3))
    total_var  = (psum_sq / b_total_pixel) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)
    return total_mean, total_std
def get_norm_dist(img, mean, std):
    dnorm_img = (img + 1) * 0.5
    sum_pixel = torch.sum(dnorm_img, dim=(0, 2, 3)) 
    b_total_pixel = dnorm_img.shape[0] * dnorm_img.shape[2] * dnorm_img.shape[3]
    x = sum_pixel/b_total_pixel
    prob_density = (math.pi*std) * torch.exp(torch.pow(-0.5*((x-mean)/std), 2))
    return prob_density
# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default, device='cuda'):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses
def run_finetune_et_model_clean_image(model, scheduler, z0, emb, 
    num_iter=1000, lr=1e-6, device="cuda:0"):
    logger.debug('Run Imagic finetune et model')
    
    alphas_cumprod = scheduler.alphas_cumprod
    with torch.no_grad():
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)
    
    model.unet.train()
    opt = torch.optim.AdamW(model.unet.parameters(), lr=lr)
    criteria = torch.nn.MSELoss()
    pbar = tqdm(range(num_iter))
    #--Additional style loss
    # import torchvision.models as models
    # cnn = models.vgg19(pretrained=True).features.to(device).eval()
    # cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    # cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    # """Run the style transfer."""
    # print('Building the style transfer model..')
    # model, style_losses, content_losses = get_style_model_and_losses(cnn,
    #     cnn_normalization_mean, cnn_normalization_std, style_img, content_img)    

    for i in pbar:
        opt.zero_grad()
        noise = torch.randn_like(z0)
        t_enc = torch.randint(1000, (1,), device=device)
        
        with torch.no_grad():
            zt = diffusion_utils.run_ddpm_q_sample(z0.clone(), t_enc, noise, 
                    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
            #--Get noisy source image
            # src_x0 = img_utils.latent2im(model, zt)
            #--Get clean image
            # src_x0 = img_utils.latent2im(model, z0)
            # tvu.save_image((src_x0 + 1) * 0.5, 'ft_debug/src_x0_noisy.png')

            zt_clean = diffusion_utils.run_ddpm_clean_q_sample(zt.clone(), t_enc, noise, 
                    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
            # #--Get noisy source image
            src_x0 = img_utils.latent2im(model, zt_clean)
            # tvu.save_image((src_x0 + 1) * 0.5, 'ft_debug/src_x0_clean.png')
            # src_mean, src_std = get_mean_std_img(src_x0)
            
            # src_dist = get_norm_dist(src_x0, src_mean, src_std)
            # tvu.save_image((src_x0 + 1) * 0.5, 'src_x0.png')
        et = model.unet(zt, t_enc, encoder_hidden_states=emb)["sample"]
        
        #--Latent image with noise
        # pred_zt =  diffusion_utils.run_ddpm_q_sample_grad(z0.clone(), t_enc, et, 
        #         sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        # pred_x0 =img_utils.latent2im_grad(model, pred_zt)
        # tvu.save_image((pred_x0 + 1) * 0.5, 'ft_debug/pred_x0_noisy.png')
        pred_z0_clean = diffusion_utils.run_ddpm_clean_q_sample(zt.clone(), t_enc, et, 
                    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        pred_x0 =img_utils.latent2im_grad(model, pred_z0_clean)
        # tvu.save_image((pred_x0 + 1) * 0.5, 'ft_debug/pred_x0_clean.png')
        # pred_mean, pred_std = get_mean_std_img(pred_x0)
        # tvu.save_image((pred_x0 + 1) * 0.5, 'pred_x0.png')
        #--Find mean and std
        loss_idn = criteria(src_x0, pred_x0)
        loss = loss_idn
        loss.backward()
        pbar.set_postfix({"loss": loss.item()})   
        opt.step()
    model.unet.eval()
# def run_finetune_et_model_mean_std_loss(model, scheduler, z0, emb, 
#     num_iter=1000, lr=1e-6, device="cuda:0"):
#     logger.debug('Run Imagic finetune et model')
    
#     alphas_cumprod = scheduler.alphas_cumprod
#     with torch.no_grad():
#         sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
#         sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)
    
#     model.unet.train()
#     # opt = torch.optim.Adam(model.unet.parameters(), lr=lr)
#     opt = torch.optim.AdamW(model.unet.parameters(), lr=lr)
#     criteria = torch.nn.MSELoss()
#     pbar = tqdm(range(num_iter))
#     #--Additional style loss
#     import torchvision.models as models
#     # cnn = models.vgg19(pretrained=True).features.to(device).eval()
#     # cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
#     # cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
#     # """Run the style transfer."""
#     # print('Building the style transfer model..')
#     # model, style_losses, content_losses = get_style_model_and_losses(cnn,
#     #     cnn_normalization_mean, cnn_normalization_std, style_img, content_img)    

#     for i in pbar:
#         opt.zero_grad()
#         noise = torch.randn_like(z0)
#         t_enc = torch.randint(1000, (1,), device=device)
        
#         with torch.no_grad():
#             zt = diffusion_utils.run_ddpm_q_sample(z0.clone(), t_enc, noise, 
#                     sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
#             #--Get noisy source image
#             # src_x0 = img_utils.latent2im(model, zt)
#             #--Get clean image
#             # src_x0 = img_utils.latent2im(model, z0)
#             # tvu.save_image((src_x0 + 1) * 0.5, 'ft_debug/src_x0_noisy.png')

#             zt_clean = diffusion_utils.run_ddpm_clean_q_sample(zt.clone(), t_enc, noise, 
#                     sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
#             # #--Get noisy source image
#             src_x0 = img_utils.latent2im(model, zt_clean)
#             # tvu.save_image((src_x0 + 1) * 0.5, 'ft_debug/src_x0_clean.png')
#             # src_mean, src_std = get_mean_std_img(src_x0)
            
#             # src_dist = get_norm_dist(src_x0, src_mean, src_std)
#             # tvu.save_image((src_x0 + 1) * 0.5, 'src_x0.png')
#         et = model.unet(zt, t_enc, encoder_hidden_states=emb)["sample"]
        
#         #--Latent image with noise
#         # pred_zt =  diffusion_utils.run_ddpm_q_sample_grad(z0.clone(), t_enc, et, 
#         #         sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
#         # pred_x0 =img_utils.latent2im_grad(model, pred_zt)
#         # tvu.save_image((pred_x0 + 1) * 0.5, 'ft_debug/pred_x0_noisy.png')
#         pred_z0_clean = diffusion_utils.run_ddpm_clean_q_sample(zt.clone(), t_enc, et, 
#                     sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
#         pred_x0 =img_utils.latent2im_grad(model, pred_z0_clean)
#         # tvu.save_image((pred_x0 + 1) * 0.5, 'ft_debug/pred_x0_clean.png')
#         # pred_mean, pred_std = get_mean_std_img(pred_x0)
#         # tvu.save_image((pred_x0 + 1) * 0.5, 'pred_x0.png')
#         #--Find mean and std
        
#         # loss_rec = criteria(et, noise)
#         loss_idn = criteria(src_x0, pred_x0)
#         # loss = loss_idn * 10.0  + loss_rec
#         loss = loss_idn
#         # loss = (loss_idn * 50.0) #+ loss_rec
#         # loss = (loss_idn * 50.0)
#         loss.backward()
#         pbar.set_postfix({"loss": loss.item()})   
#         opt.step()
#     model.unet.eval()

def run_finetune_et_model_idn_loss(model, scheduler, z0, emb, 
    num_iter=1000, lr=1e-6, device="cuda:0"):
    logger.debug('Run Imagic finetune et model')
    
    alphas_cumprod = scheduler.alphas_cumprod
    with torch.no_grad():
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)
    
    model.unet.train()
    # opt = torch.optim.Adam(model.unet.parameters(), lr=lr)
    opt = torch.optim.AdamW(model.unet.parameters(), lr=lr)
    criteria = torch.nn.MSELoss()
    pbar = tqdm(range(num_iter))
    
    for i in pbar:
        opt.zero_grad()
        noise = torch.randn_like(z0)
        t_enc = torch.randint(1000, (1,), device=device)
        zt = diffusion_utils.run_ddpm_q_sample(z0.clone(), t_enc, noise, 
                sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        et = model.unet(zt, t_enc, encoder_hidden_states=emb)["sample"]
        
        #--Latent image with noise
        pred_zt =  diffusion_utils.run_ddpm_q_sample_grad(z0.clone(), t_enc, et, 
                sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        loss_rec = criteria(et, noise)
        loss_idn = criteria(zt, pred_zt)
        loss = (loss_idn * 2.0) + loss_rec
        # loss = (loss_idn * 50.0)
        loss.backward()
        pbar.set_postfix({"loss": loss.item()})   
        opt.step()
    model.unet.eval()

def run_finetune_et_model_cycle_loss(
            model, scheduler, A_z0, B_z0, 
            emb, num_iter=1000, lr=1e-6, 
            device="cuda:0"):
    logger.debug('Run finetune et model with cycle')
    alphas_cumprod = scheduler.alphas_cumprod
    with torch.no_grad():
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)
    
    model.unet.train()
    # opt = torch.optim.Adam(model.unet.parameters(), lr=lr)
    opt = torch.optim.AdamW(model.unet.parameters(), lr=lr)
    criteria = torch.nn.MSELoss()
    pbar = tqdm(range(num_iter))
    
    for i in pbar:
        opt.zero_grad()
        noise = torch.randn_like(A_z0)
        t_enc = torch.randint(1000, (1,), device=device)
        
        A_zt = diffusion_utils.run_ddpm_q_sample(A_z0.clone(), t_enc, noise, 
                sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        B_zt = diffusion_utils.run_ddpm_q_sample(B_z0.clone(), t_enc, noise, 
                sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        with torch.no_grad():
            B_et = model.unet(B_zt, t_enc, encoder_hidden_states=emb)["sample"]
        
        A_et = model.unet(A_zt, t_enc, encoder_hidden_states=emb)["sample"]
        
        loss_rec = criteria(A_et, noise)
        loss_A2B = criteria(A_et, B_et)
        loss = loss_rec + loss_A2B
        # loss = loss_A2B
        loss.backward()
        pbar.set_postfix({"loss": loss.item()})   
        opt.step()
    model.unet.eval()
def run_finetune_et_model_with_syntext(model, scheduler, data, 
    num_iter=1000, lr=1e-6, device="cuda:0"):
    logger.debug('Run Imagic finetune et model')
    alphas_cumprod = scheduler.alphas_cumprod
    with torch.no_grad():
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)
    model.unet.train()
    # opt = torch.optim.Adam(model.unet.parameters(), lr=lr)
    opt = torch.optim.AdamW(model.unet.parameters(), lr=lr)
    criteria = torch.nn.MSELoss()
    pbar = tqdm(range(num_iter))
    for i in pbar:
        idx = random.randint(0, len(data)-1)
        prep_data = data[idx]
        z0 = prep_data['tgt_z0'].to(device)
        emb = prep_data['tgt_cond_emb'].to(device)
        
        opt.zero_grad()
        noise = torch.randn_like(z0)
        t_enc = torch.randint(1000, (1,), device=device)
        zt = diffusion_utils.run_ddpm_q_sample(z0.clone(), t_enc, noise, 
                sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        et = model.unet(zt, t_enc, encoder_hidden_states=emb)["sample"]
        loss = criteria(et, noise)
        loss.backward()
        pbar.set_postfix({"loss": loss.item()})   
        opt.step()
    model.unet.eval()
    # torch.save(model.unet.state_dict(), 'data/ft_et_text_syntext4chars.ckpt')
    torch.save(model.unet.state_dict(), 'data/ft_et_text_syntext4chars_20k.ckpt')
    # torch.save(model.unet.state_dict(), 'data/ft_et_text.ckpt')
@torch.no_grad()
def prepare_input(ldm_stable, tokenizer, img_id, text, src_dir, 
                target_size=256, MAX_NUM_WORDS=77, device='cuda'):
    g = torch.Generator(device=device).manual_seed(888)
    img_path = os.path.join(src_dir, img_id)
    img = img_utils.load_img(img_path, target_size=target_size).to(device).unsqueeze(0)
    # tvu.save_image((img + 1) * 0.5, 'test.jpg')
    z0 = img_utils.im2latent(ldm_stable, img, g)
    uncond_emb, cond_emb = prompt_utils.gen_init_prompt_to_emb(ldm_stable, tokenizer, 
                                [text], MAX_NUM_WORDS, device=device)
    return z0, cond_emb

def run_finetune_et_model_with_syntext_from_path(model, tokenizer, scheduler, texts_list, 
    num_iter=1000, lr=1e-6, dataset_dir ='verihubs-2TB', device="cuda:0"):
    logger.debug('Run Imagic finetune et model')
    alphas_cumprod = scheduler.alphas_cumprod
    with torch.no_grad():
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)
    model.unet.train()
    # opt = torch.optim.Adam(model.unet.parameters(), lr=lr)
    opt = torch.optim.AdamW(model.unet.parameters(), lr=lr)
    criteria = torch.nn.MSELoss()
    pbar = tqdm(range(num_iter))
    
    for i in pbar:
        img_id = random.choice(list(texts_list))
        # src_prompt = f"A text that reads '{texts_list[img_id]['src_text']}'"
        tgt_prompt = f"A text that reads '{texts_list[img_id]['tgt_text']}'"
        z0, emb = prepare_input(model, tokenizer, img_id, tgt_prompt, dataset_dir)
        # z0 = prep_data['tgt_z0'].to(device)
        # emb = prep_data['tgt_cond_emb'].to(device)
        opt.zero_grad()
        noise = torch.randn_like(z0)
        t_enc = torch.randint(1000, (1,), device=device)
        zt = diffusion_utils.run_ddpm_q_sample(z0.clone(), t_enc, noise, 
                sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        et = model.unet(zt, t_enc, encoder_hidden_states=emb)["sample"]
        loss = criteria(et, noise)
        loss.backward()
        pbar.set_postfix({"loss": loss.item()})   
        opt.step()
    model.unet.eval()
    # torch.save(model.unet.state_dict(), 'data/ft_et_text_syntext4chars.ckpt')
    torch.save(model.unet.state_dict(), 'data/ft_et_text_syntext345chars_1M.ckpt')
    # torch.save(model.unet.state_dict(), 'data/ft_et_text.ckpt')
@torch.no_grad()
def prepare_input_from_input_img(ldm_stable, img, 
                target_size=256, MAX_NUM_WORDS=77, device='cuda'):
    g = torch.Generator(device=device).manual_seed(888)
    image = img_utils.augment_and_norm_image(img).to(device).unsqueeze(0)
    # tvu.save_image((image + 1) * 0.5, 'test.jpg')
    z0 = img_utils.im2latent(ldm_stable, image, g)
    
    return z0
def run_finetune_et_model_input_image(model, scheduler, src_img, 
    src_cond_emb,
    num_iter=1000, lr=1e-6, dataset_dir ='verihubs-2TB', device="cuda:0"):
    logger.debug('Run Imagic finetune et model')
    alphas_cumprod = scheduler.alphas_cumprod
    with torch.no_grad():
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)
    model.unet.train()
    # opt = torch.optim.Adam(model.unet.parameters(), lr=lr)
    opt = torch.optim.AdamW(model.unet.parameters(), lr=lr)
    criteria = torch.nn.MSELoss()
    pbar = tqdm(range(num_iter))
    
    for i in pbar:
        z0 = prepare_input_from_input_img(model, src_img.copy())
        emb = src_cond_emb.clone()
        noise = torch.randn_like(z0)
        t_enc = torch.randint(1000, (1,), device=device)
        zt = diffusion_utils.run_ddpm_q_sample(z0.clone(), t_enc, noise, 
                sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        
        opt.zero_grad(set_to_none=True)
        et = model.unet(zt, t_enc, encoder_hidden_states=emb)["sample"]
        loss = criteria(et, noise)
        loss.backward()
        pbar.set_postfix({"loss": loss.item()})   
        opt.step()

    model.unet.eval()
    # torch.save(model.unet.state_dict(), 'data/ft_et_text_syntext4chars.ckpt')
    # torch.save(model.unet.state_dict(), 'data/ft_et_ic13_142.ckpt')
    torch.save(model.unet.state_dict(), 'data/ft_et_ic13_142_5k.ckpt')

def run_finetune_et_model_input_image_specific_augm(model, scheduler, src_img, 
    src_cond_emb,
    num_iter=1000, lr=1e-6, dataset_dir ='verihubs-2TB', device="cuda:0"):
    logger.debug('Run Imagic finetune et model')
    alphas_cumprod = scheduler.alphas_cumprod
    with torch.no_grad():
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)
    model.unet.train()
    # opt = torch.optim.Adam(model.unet.parameters(), lr=lr)
    opt = torch.optim.AdamW(model.unet.parameters(), lr=lr)
    criteria = torch.nn.MSELoss()
    pbar = tqdm(range(num_iter))
    
    for i in pbar:
        z0 = prepare_input_from_input_img(model, src_img.copy())
        emb = src_cond_emb.clone()
        noise = torch.randn_like(z0)
        t_enc = torch.randint(1000, (1,), device=device)
        zt = diffusion_utils.run_ddpm_q_sample(z0.clone(), t_enc, noise, 
                sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        
        opt.zero_grad(set_to_none=True)
        et = model.unet(zt, t_enc, encoder_hidden_states=emb)["sample"]
        loss = criteria(et, noise)
        loss.backward()
        pbar.set_postfix({"loss": loss.item()})   
        opt.step()

    model.unet.eval()
    # torch.save(model.unet.state_dict(), 'data/ft_et_text_syntext4chars.ckpt')
    # torch.save(model.unet.state_dict(), 'data/ft_et_ic13_142.ckpt')
    torch.save(model.unet.state_dict(), 'data/ft_et_ic13_142_5k.ckpt')