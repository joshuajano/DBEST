import importlib
import logging
from loguru import logger
import os 
import PIL
import numpy as np
import cv2
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
from fastai.vision import *
import torch
import torchvision.utils as tvu
import torch.nn as nn
import torchvision.models as models

from t2tldm.utils.text_recog_utils import Config, CharsetMapper
from t2tldm.losses.text_recog_loss import MultiLosses
from t2tldm.losses.style_transfer_loss import build_vgg_loss
from t2tldm.utils import img_utils, diffusion_utils, prompt_utils, denoising_utils
content_layers_default = ['conv_4'],
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'],

def onehot(label, depth, device=None):
    """ 
    Args:
        label: shape (n1, n2, ..., )
        depth: a scalar
    Returns:
        onehot: (n1, n2, ..., depth)
    """
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label, device=device)
    onehot = torch.zeros(label.size() + torch.Size([depth]), device=device)
    onehot = onehot.scatter_(-1, label.unsqueeze(-1), 1)

    return onehot
def get_model(config):
    import importlib
    names = config.model_name.split('.')
    module_name, class_name = '.'.join(names[:-1]), names[-1]
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls(config)
    logging.info(model)
    model = model.eval()
    return model
def load(model, file, device=None, strict=True):
    if device is None: device = 'cpu'
    elif isinstance(device, int): device = torch.device('cuda', device)
    assert os.path.isfile(file)
    state = torch.load(file, map_location=device)
    if set(state.keys()) == {'model', 'opt'}:
        state = state['model']
    model.load_state_dict(state, strict=strict)
    return model
def preprocess(img, width, height):
    img = cv2.resize(np.array(img), (width, height))
    img = transforms.ToTensor()(img).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])
    return (img-mean[...,None,None]) / std[...,None,None]
# @torch.no_grad()
def normalize_img_style_transfer(x0, device='cuda'):
    #--Denorm to 0 - 1
    img = (x0 + 1) * 0.5
    #--Resize
    # transform = transforms.Resize(size = (width, height))
    # img = transform(img)
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).to(device)

    return (img-mean[...,None,None]) / std[...,None,None]
def normalize_text_recognition(x0, width=128, height=32, device='cuda'):
    #--Denorm to 0 - 1
    img = (x0 + 1) * 0.5
    #--Resize
    transform = transforms.Resize(size = (width, height))
    img = transform(img)
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).to(device)

    return (img-mean[...,None,None]) / std[...,None,None]
def postprocess(output, charset, model_eval):

    def _get_output(last_output, model_eval):
        if isinstance(last_output, (tuple, list)): 
            for res in last_output:
                if res['name'] == model_eval: output = res
        else: output = last_output
        return output

    def _decode(logit):
        """ Greed decode """
        out = F.softmax(logit, dim=2)
        pt_text, pt_scores, pt_lengths = [], [], []
        for o in out:
            text = charset.get_text(o.argmax(dim=1), padding=False, trim=False)
            text = text.split(charset.null_char)[0]  # end at end-token
            pt_text.append(text)
            pt_scores.append(o.max(dim=1)[0])
            pt_lengths.append(min(len(text) + 1, charset.max_length))  # one for end-token
        return pt_text, pt_scores, pt_lengths

    output = _get_output(output, model_eval)
    logits, pt_lengths = output['logits'], output['pt_lengths']
    pt_text, pt_scores, pt_lengths_ = _decode(logits)
    
    return pt_text, pt_scores, pt_lengths_
def run_init_text_recog(config, checkpoints_path, device='cuda'):
    logger.info(f'Load pretrained text recognition from {checkpoints_path}')
    config.global_phase = 'test'
    config.model_vision_checkpoint, config.model_language_checkpoint = None, None
    model = get_model(config).to(device)
    config.model_checkpoint=checkpoints_path
    model = load(model, config.model_checkpoint, device=device)
    logger.info(f'Load pretrained text recognition from {config.dataset_charset_path}')
    charset = CharsetMapper(filename=config.dataset_charset_path,
                            max_length=config.dataset_max_length + 1)
    config.model_eval = 'alignment'
    logger.success(f'Succesfully load pretrained text recogntion')
    return model, charset
def run_recognition_model(ldm_stable, txtrec_model, z0 ):
    x0 =img_utils.latent2im(ldm_stable, z0)
    img = normalize_text_recognition(x0, width=32, height=128)
    res = txtrec_model(img)
    return res
def optimize_emb_with_text_recog_loss(ldm_stable, scheduler, zT, uncond_emb, 
                                    cond_emb, target_text,
                                    yaml_file = '/home/verihubs/Documents/joshua/ABINet/configs/train_abinet.yaml',
                                    checkpoints_path ='/home/verihubs/Documents/joshua/ABINet/workdir/workdir/train-abinet/best-train-abinet.pth', 
                                    num_iter = 100,
                                    NUM_DDIM_STEPS =50,
                                    start_time = 50,  
                                    guidance_scale = 5.,
                                    lr=0.0001, step_recog_loss = 25, #lr=0.0001,
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
    #--Learnable condition embedding    
    emb = cond_emb.detach().clone()
    emb.requires_grad = True
    # opt = torch.optim.Adam([emb], lr=lr)
    opt = torch.optim.AdamW([emb], lr=lr)
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
    # opt = torch.optim.SGD([emb], lr=lr)
    criteria = MultiLosses(one_hot=True)
    length = tensor(len(target_text) + 1).to(dtype=torch.long).unsqueeze(0) 
    label = charset.get_labels(target_text, case_sensitive=False)
    label = tensor(label).to(dtype=torch.long)
    onehot_label = onehot(label, charset.num_classes).unsqueeze(0).to(device)
    pbar = tqdm(range(num_iter))
    best_opt_emb = None
    best_loss = 0
    for i in pbar:
        zt = zT.clone()
        context = torch.cat([uncond_emb, emb])
        ldm_stable.scheduler.set_timesteps(NUM_DDIM_STEPS)
        with tqdm(total=NUM_DDIM_STEPS, desc=f"DDIM generative process {i}") as progress_bar:
            for i, t in enumerate(ldm_stable.scheduler.timesteps[-start_time:]):
                # context = torch.cat([uncond_emb, emb])
                zt_cat = torch.cat([zt] * 2)
                if i < step_recog_loss:
                    with torch.no_grad():
                        noise_pred = ldm_stable.unet(zt_cat, t, encoder_hidden_states=context.clone())["sample"]
                        et_uncond, et_cond = noise_pred.chunk(2)
                        et = et_uncond + guidance_scale * (et_cond - et_uncond) /\
                            torch.norm(et_cond - et_uncond) * torch.norm(et_uncond)
                        zt = ldm_stable.scheduler.step(et, t, zt)["prev_sample"]
                else:
                    with torch.enable_grad():
                        noise_pred = ldm_stable.unet(zt_cat, t, encoder_hidden_states=context.clone())["sample"]
                        et_uncond, et_cond = noise_pred.chunk(2)

                        et = et_uncond + guidance_scale * (et_cond - et_uncond) /\
                            torch.norm(et_cond - et_uncond) * torch.norm(et_uncond)
                        zt = ldm_stable.scheduler.step(et, t, zt)["prev_sample"]
                    
                    res =run_recognition_model(ldm_stable, txtrec_model, zt)
                    pt_text, _, __ = postprocess(res, charset, config.model_eval)
                    
                    opt.zero_grad(set_to_none=True)
                    loss = criteria(res, onehot_label, length)
                    if best_opt_emb == None:
                        best_opt_emb = emb
                        best_loss = loss
                    if loss < best_loss and pt_text[0] == target_text.lower():
                        best_opt_emb = emb
                        best_loss = loss
                    if pt_text[0] == target_text.lower() and loss<=3.:
                        if use_lr_decay:
                            if lr_type =='rop':
                                lr_scheduler.step(loss)
                            else:
                                lr_scheduler.step()
                    
                    loss.backward()
                    opt.step()
                    if use_lr_decay:
                        if lr_type =='rop':
                            lr_scheduler.step(loss)
                        else:
                            lr_scheduler.step()
                    zt = zt.detach().clone()
                    progress_bar.set_postfix({"loss": loss.item(), "text": pt_text})   
                tvu.save_image((zt + 1) * 0.5, os.path.join(save_path, f'z{i}.png'))
                progress_bar.update(1)
            # if best_opt_emb == None:
            #     best_opt_emb = emb
            #     best_loss = loss
            # if loss < best_loss and pt_text[0] == target_text.lower():
            #     best_opt_emb = emb
            #     best_loss = loss
        # if pt_text[0] == target_text.lower() and loss<=1.:
        #     break
    x0 =img_utils.latent2im(ldm_stable, zt)
    tvu.save_image((x0 + 1) * 0.5, os.path.join(save_path, f'x0_rec_mixture_opt_source_emb.png'))
    return best_opt_emb
    # return emb
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    # def forward(self, input):
    #     self.loss = F.mse_loss(input, self.target)
    #     return input
    def forward(self, input):
        loss = F.mse_loss(input, self.target)
        return loss
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    # def forward(self, input):
    #     G = gram_matrix(input)
    #     self.loss = F.mse_loss(G, self.target)
    #     return input
    def forward(self, input):
        G = gram_matrix(input)
        # self.loss = F.mse_loss(G, self.target)
        loss = F.mse_loss(G, self.target)
        return loss
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
class Vgg19(torch.nn.Module):
    def __init__(self):
        
        super(Vgg19, self).__init__()
        features = list(models.vgg19(pretrained = True).features)
        self.features = torch.nn.ModuleList(features).eval()
        
    def forward(self, x):
        
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            
            if ii in {1, 6, 11, 20, 29}:
                results.append(x)
        return results
def run_finetune_et_model_text_recog_style(model, scheduler, z0, emb, target_text,
    yaml_file = '/home/verihubs/Documents/joshua/ABINet/configs/train_abinet.yaml',
    checkpoints_path ='/home/verihubs/Documents/joshua/ABINet/workdir/workdir/train-abinet/best-train-abinet.pth', 
    num_iter=1000, 
    lambda_style= 1000000,
    lr=1e-6,
    # content_layers_default = ['conv_4'],
    # style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'],
    
    device="cuda:0"):
    vgg_features = Vgg19().to(device)
    with torch.no_grad():
        x0 = img_utils.latent2im(model, z0.detach())
        real_img = normalize_img_style_transfer(x0)

    #--Style and content check
    # cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    # cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    # cnn = models.vgg19(pretrained=True).features.to(device).eval()
    # stl_x0 = img_utils.latent2im(model, z0.detach())
    # stl_x0 = normalize_img_style_transfer(stl_x0)

    logger.debug('Run finetune et model text recognition + style')
    config = Config(yaml_file)
    txtrec_model, charset = run_init_text_recog(config, checkpoints_path, device)
    alphas_cumprod = scheduler.alphas_cumprod
    with torch.no_grad():
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)
    model.unet.train()
   
    opt = torch.optim.AdamW(model.unet.parameters(), lr=lr)
    criteria = MultiLosses(one_hot=True)
    length = tensor(len(target_text) + 1).to(dtype=torch.long).unsqueeze(0) 
    label = charset.get_labels(target_text, case_sensitive=False)
    label = tensor(label).to(dtype=torch.long)
    onehot_label = onehot(label, charset.num_classes).unsqueeze(0).to(device)
    
    pbar = tqdm(range(num_iter))
    for i in pbar:
        opt.zero_grad(set_to_none=None)
        noise = torch.randn_like(z0)
        t_enc = torch.randint(low=500, high=1000, size=(1,), device=device)
        with torch.no_grad():
            zt = diffusion_utils.run_ddpm_q_sample(z0.clone(), t_enc, noise, 
                    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        et = model.unet(zt, t_enc, encoder_hidden_states=emb)["sample"]
        pred_z0_clean = diffusion_utils.run_ddpm_clean_q_sample(zt.clone(), t_enc, et, 
                    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        
        pred_x0 =img_utils.latent2im_grad(model, pred_z0_clean)
        pred_img_tr = normalize_text_recognition(pred_x0, width=32, height=128, device='cpu')

        #--Compute loss
        pred_img = normalize_img_style_transfer(pred_x0)
        i_vgg = torch.cat((real_img.clone(), pred_img), dim = 0)
        out_vgg = vgg_features(i_vgg)
        l_f_vgg_per, l_f_vgg_style = build_vgg_loss(out_vgg)
        res = txtrec_model(pred_img_tr)
        
        loss = criteria(res, onehot_label, length) * (t_enc / 1000)  + (l_f_vgg_style * lambda_style)
        loss.backward()
        pbar.set_postfix({"loss": loss.item()})   
        opt.step()
    model.unet.eval()
def run_finetune_et_model_text_recog(model, scheduler, z0, emb, target_text,
    yaml_file = '/home/verihubs/Documents/joshua/ABINet/configs/train_abinet.yaml',
    checkpoints_path ='/home/verihubs/Documents/joshua/ABINet/workdir/workdir/train-abinet/best-train-abinet.pth', 
    num_iter=1000, 
    lr=1e-6, device="cuda:0"):

    logger.debug('Run finetune et model text recognition')
    config = Config(yaml_file)
    txtrec_model, charset = run_init_text_recog(config, checkpoints_path, device)
    alphas_cumprod = scheduler.alphas_cumprod
    with torch.no_grad():
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)
    model.unet.train()
   
    opt = torch.optim.AdamW(model.unet.parameters(), lr=lr)
    criteria = MultiLosses(one_hot=True)
    length = tensor(len(target_text) + 1).to(dtype=torch.long).unsqueeze(0) 
    label = charset.get_labels(target_text, case_sensitive=False)
    label = tensor(label).to(dtype=torch.long)
    onehot_label = onehot(label, charset.num_classes).unsqueeze(0).to(device)

    pbar = tqdm(range(num_iter))
    for i in pbar:
        opt.zero_grad(set_to_none=None)
        noise = torch.randn_like(z0)
        t_enc = torch.randint(low=500, high=1000, size=(1,), device=device)
        with torch.no_grad():
            zt = diffusion_utils.run_ddpm_q_sample(z0.clone(), t_enc, noise, 
                    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        et = model.unet(zt, t_enc, encoder_hidden_states=emb)["sample"]
        pred_z0_clean = diffusion_utils.run_ddpm_clean_q_sample(zt.clone(), t_enc, et, 
                    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        
        pred_x0 =img_utils.latent2im_grad(model, pred_z0_clean)
        img = normalize_text_recognition(pred_x0, width=32, height=128)
        res = txtrec_model(img)
        
        loss = criteria(res, onehot_label, length) * (t_enc / 1000) 
        loss.backward()
        pbar.set_postfix({"loss": loss.item()})   
        opt.step()
    model.unet.eval()

def opt_emb_DDPM_text_recog(model, scheduler, 
                            z0, src_emb, 
                            cond_emb, target_text,
                            yaml_file = '/home/verihubs/Documents/joshua/ABINet/configs/train_abinet.yaml',
                            checkpoints_path ='/home/verihubs/Documents/joshua/ABINet/workdir/workdir/train-abinet/best-train-abinet.pth', 
                            num_iter = 100,
                            lr=0.0001, 
                            save_dir = '',
                            device='cpu'):
    save_path = os.path.join(save_dir, 'generative')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    config = Config(yaml_file)
    txtrec_model, charset = run_init_text_recog(config, checkpoints_path, device)
    
    emb = cond_emb.detach().clone()
    emb.requires_grad = True
    opt = torch.optim.AdamW([emb], lr=lr)
    criteria = MultiLosses(one_hot=True)
    length = tensor(len(target_text) + 1).to(dtype=torch.long).unsqueeze(0) 
    label = charset.get_labels(target_text, case_sensitive=False)
    label = tensor(label).to(dtype=torch.long)
    onehot_label = onehot(label, charset.num_classes).unsqueeze(0).to(device)
    pbar = tqdm(range(num_iter))

    alphas_cumprod = scheduler.alphas_cumprod
    with torch.no_grad():
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)

    for i in pbar:
        opt.zero_grad(set_to_none=None)
        noise = torch.randn_like(z0)
        # t_enc = torch.randint(1000, (1,), device=device)
        t_enc = torch.randint(low=500, high=1000, size=(1,), device=device)
        #--Get ground truth
        with torch.no_grad():
            #--Try noise from et 
            # noise = model.unet(z0.clone(), t_enc, encoder_hidden_states=src_emb.detach())["sample"]    
            zt = diffusion_utils.run_ddpm_q_sample(z0.clone(), t_enc, noise, 
                    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        et = model.unet(zt, t_enc, encoder_hidden_states=emb)["sample"]
        pred_z0_clean = diffusion_utils.run_ddpm_clean_q_sample(zt.clone(), t_enc, et, 
                    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        
        pred_x0 =img_utils.latent2im_grad(model, pred_z0_clean)
        # tvu.save_image((pred_x0 + 1) * 0.5, 'ft_debug/pred_x0_clean.png')
        # img = normalize_text_recognition(pred_x0, width=32, height=128, device='cpu')
        img = normalize_text_recognition(pred_x0, width=32, height=128, device=device)
        res = txtrec_model(img)
        # pt_text, _, __ = postprocess(res, charset, config.model_eval)
        loss = criteria(res, onehot_label, length) * (t_enc / 1000) 
        loss.backward()
        opt.step()
    return emb.detach()

def opt_emb_DDPM_text_recog_style(model, scheduler, 
                            z0, src_emb, 
                            cond_emb, target_text,
                            yaml_file = '/home/verihubs/Documents/joshua/ABINet/configs/train_abinet.yaml',
                            checkpoints_path ='/home/verihubs/Documents/joshua/ABINet/workdir/workdir/train-abinet/best-train-abinet.pth', 
                            num_iter = 100,
                            lr=0.0001, 
                            lambda_style =1000000,
                            save_dir = '',
                            device='cuda'):
    save_path = os.path.join(save_dir, 'generative')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    config = Config(yaml_file)
    txtrec_model, charset = run_init_text_recog(config, checkpoints_path, device)
    vgg_features = Vgg19().to(device)
    with torch.no_grad():
        x0 = img_utils.latent2im(model, z0.detach())
        real_img = normalize_img_style_transfer(x0)

    emb = cond_emb.detach().clone()
    emb.requires_grad = True
    opt = torch.optim.AdamW([emb], lr=lr)
    criteria = MultiLosses(one_hot=True)
    length = tensor(len(target_text) + 1).to(dtype=torch.long).unsqueeze(0) 
    label = charset.get_labels(target_text, case_sensitive=False)
    label = tensor(label).to(dtype=torch.long)
    onehot_label = onehot(label, charset.num_classes).unsqueeze(0).to(device)
    pbar = tqdm(range(num_iter))

    alphas_cumprod = scheduler.alphas_cumprod
    with torch.no_grad():
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)

    for i in pbar:
        opt.zero_grad(set_to_none=None)
        noise = torch.randn_like(z0)
        # t_enc = torch.randint(1000, (1,), device=device)
        t_enc = torch.randint(low=500, high=1000, size=(1,), device=device)
        #--Get ground truth
        with torch.no_grad():
            #--Try noise from et 
            # noise = model.unet(z0.clone(), t_enc, encoder_hidden_states=src_emb.detach())["sample"]    
            zt = diffusion_utils.run_ddpm_q_sample(z0.clone(), t_enc, noise, 
                    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        et = model.unet(zt, t_enc, encoder_hidden_states=emb)["sample"]
        pred_z0_clean = diffusion_utils.run_ddpm_clean_q_sample(zt.clone(), t_enc, et, 
                    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        
        pred_x0 =img_utils.latent2im_grad(model, pred_z0_clean)
        pred_img_tr = normalize_text_recognition(pred_x0, width=32, height=128)

        #--Compute loss
        pred_img = normalize_img_style_transfer(pred_x0)
        i_vgg = torch.cat((real_img.clone(), pred_img), dim = 0)
        out_vgg = vgg_features(i_vgg)
        l_f_vgg_per, l_f_vgg_style = build_vgg_loss(out_vgg)
        img = normalize_text_recognition(pred_img_tr, width=32, height=128)
        res = txtrec_model(img)
        # pt_text, _, __ = postprocess(res, charset, config.model_eval)
        loss =( criteria(res, onehot_label, length) * (t_enc / 1000)) + (l_f_vgg_style * lambda_style)
        loss.backward()
        opt.step()
    return emb.detach()