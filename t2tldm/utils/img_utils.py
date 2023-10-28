from PIL import Image
from torchvision import transforms
import torch 
import albumentations as A
import cv2
import numpy as np
import torchvision
from t2tldm.utils import prompt_utils, diffusion_utils
from torchvision.transforms import functional as F
augment_transform = A.Compose(
                [  
                    A.HorizontalFlip(p=0.5), 
                    A.RandomScale(p=0.3),
                    A.Rotate(p=0.5),
                    # A.RandomBrightnessContrast(p=0.2),
                    # A.ChannelShuffle(p=0.3), 
                    # A.Blur(p=0.4),
                    # A.JpegCompression(p=0.3),
                    A.RandomRotate90(p=0.3), 
                    # A.OpticalDistortion(always_apply=True),
                    # A.ElasticTransform(always_apply=True),
                ])
tform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
        ])
def im2latent(pipe, x0, generator):
    try:
        init_latent_dist = pipe.vae.encode(x0).latent_dist
    except:
        init_latent_dist = pipe.vqvae.encode(x0).latent_dist
    init_latents = init_latent_dist.sample(generator=generator)
    return init_latents * 0.18215
# @torch.no_grad()
def latent2im(pipe, zt):
    latents = 1 / 0.18215 * zt
    try:
        image = pipe.vae.decode(latents)['sample']
    except:
        image = pipe.vqvae.decode(latents)['sample']
    return image
    
def latent2im_grad(pipe, zt):
    latents = 1 / 0.18215 * zt
    try:
        image = pipe.vae.decode(latents)['sample']
    except:
        image = pipe.vqvae.decode(latents)['sample']
    return image
def color_transfer(s_img, t_img):
    # s_img = cv2.imread(src_path)
    s = cv2.cvtColor(s_img, cv2.COLOR_BGR2LAB)
    # t_img = cv2.imread(tgt_path)
    t = cv2.cvtColor(t_img, cv2.COLOR_BGR2LAB)
    s_mean, s_std = get_mean_and_std(s)
    t_mean, t_std = get_mean_and_std(t)
    height, width, channel = s.shape
    for i in range(0, height):
        for j in range(0,width):
            for k in range(0, channel):
                x = s[i,j,k]
                x = ((x-s_mean[k])*(t_std[k]/s_std[k]))+t_mean[k]
                x = round(x)
                x = 0 if x<0 else x
                x = 255 if x>255 else x
                s[i,j,k] = x
    s = cv2.cvtColor(s,cv2.COLOR_LAB2BGR)
    # cv2.imwrite(src_path[:-4] + '_ct.png', s)
    return s 

def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")
    image = image.resize((target_size, target_size))
    
    tform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])
    image = tform(image)
    return 2.*image - 1.
def load_img_gradio(image, target_size=512):
    """Load an image, resize and output -1..1"""
    # image = Image.open(path).convert("RGB")
    # image = image.resize((target_size, target_size))
    
    tform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])
    image = tform(image)
    return 2.*image - 1.
def get_mean_and_std(x):
	x_mean, x_std = cv2.meanStdDev(x)
	x_mean = np.hstack(np.around(x_mean,2))
	x_std = np.hstack(np.around(x_std,2))
	return x_mean, x_std
def load_img_dynamic_norm(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = cv2.imread(path)
    image = cv2.resize(image, (target_size, target_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    np_image = np.array(image).astype(np.float32)
    np_image /= 255. 

    pixel_sum = np_image.sum(axis=(0, 1))
    count = 256 * 256
    total_mean = pixel_sum / count
    psum_sq = (np_image ** 2).sum(axis=(0, 1))
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = np.sqrt(total_var)
    
    # image = (image - image.min()) / (image.max() - image.min())
    # np_image = np.array(image).astype(np.float32)
    # np_image /= 255. 
    # norm_img = np.zeros(np_image.shape)
    # mean_r = np_image[..., 0].mean()
    # std_r = np_image[..., 0].std()
    # mean_g = np_image[..., 1].mean()
    # std_g = np_image[..., 1].std()
    # mean_b = np_image[..., 2].mean()
    # std_b = np_image[..., 2].std()
    # norm_img[..., 0] = (np_image[..., 0] - mean_r) / std_r
    # norm_img[..., 1] = (np_image[..., 1] - mean_g) / std_g
    # norm_img[..., 2] = (np_image[..., 2] - mean_b) / std_b
    
    # z = (np_image - np_image.mean(axis=(0, 1, 2), keepdims=True)) / np_image.std(axis=(0,1,2), keepdims=True)
    # orig_img = image.copy()
    # image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    # s_mean, s_std = get_mean_and_std(image)
    # height, width, channel = image.shape
    # placeholder = np.zeros((height, width, channel))
    # # for k in range(0, channel):
    # #     image[:, :, k] = (image[:, :, k] - s_mean[k]) / s_std[k]
    # # image = image.resize((target_size, target_size))
    # for i in range(0, height):
    #     pass
    #     for j in range(0, width):
    #         pass
    #         for k in range(0, channel):
    #             x = image[i,j,k]
    #             x = ((x-s_mean[k]) / s_std[k])
    #             placeholder[i,j,k] = x

    # image = cv2.cvtColor(placeholder, cv2.COLOR_LAB2BGR)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])
    
    image = torch.from_numpy(np_image).float()
    image = torch.permute(image, (2, 0, 1))
    norm = F.normalize(image, mean=total_mean, std=total_std)
    
    # return 2.*image - 1.
    return norm, total_mean, total_std
def load_np_image(path, target_size=512):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (target_size, target_size))
    return image

def augment_and_norm_image(img):
    augm_img = augment_transform(image=img)
    #--Convert to pil image
    image = Image.fromarray(augm_img["image"])
    image = tform(image)
    return 2.*image - 1.
def set_augment_and_norm_image(img):
    augment_transform = A.Compose(
                [ 
                    # A.IAAPiecewiseAffine(p=1.),
                    # A.HorizontalFlip(p=1.),  
                    # A.RandomScale(p=1.),
                    # A.Rotate(p=1.),
                ])
    augm_img = augment_transform(image=img)
    #--Convert to pil image
    image = Image.fromarray(augm_img["image"])
    image = tform(image)
    return 2.*image - 1.

@torch.no_grad()
def prepare_input(ldm_stable, scheduler, context,
                img_path,seed,  
                ddim_step = 50, 
                img_size =256,
                save_path = 'debug_images/', 
                device='cuda'):
    img = load_img(img_path, target_size=img_size).to(device).unsqueeze(0)
    z0 = im2latent(ldm_stable, img, seed)
    
    _, zT = diffusion_utils.run_ddim_q_sample(ldm_stable, scheduler, z0.clone(), context.clone(), 
                ddim_step, COND_Q_SAMPLE=True, save_dir=save_path)
    return z0, zT
@torch.no_grad()
def prepare_input_gradio(ldm_stable, scheduler, context,
                image,seed,  
                ddim_step = 50, 
                img_size =256,
                save_path = 'debug_images/', 
                device='cuda'):
    img = load_img_gradio(image, target_size=img_size).to(device).unsqueeze(0)
    z0 = im2latent(ldm_stable, img, seed)
    
    _, zT = diffusion_utils.run_ddim_q_sample(ldm_stable, scheduler, z0.clone(), context.clone(), 
                ddim_step, COND_Q_SAMPLE=True, save_dir=save_path)
    return z0, zT
def prepare_input_dynamic_norm_img(ldm_stable, scheduler, context,
                img_path,seed,  
                ddim_step = 50, 
                img_size =256,
                save_path = 'debug_images/', 
                device='cuda'):
    img, total_mean, total_std = load_img_dynamic_norm(img_path, target_size=img_size)
    
    # mean = torch.from_numpy(total_mean).float().to(device).unsqueeze(0)
    # std = torch.from_numpy(total_std).float().to(device).unsqueeze(0)
    img = img.to(device).unsqueeze(0)
    # img = load_img(img_path, target_size=img_size).to(device).unsqueeze(0)
    z0 = im2latent(ldm_stable, img, seed)
    
    _, zT = diffusion_utils.run_ddim_q_sample(ldm_stable, scheduler, z0.clone(), context.clone(), 
                ddim_step, COND_Q_SAMPLE=True, save_dir=save_path)
    return z0, zT, total_mean, total_std