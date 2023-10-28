from loguru import logger
import torch
from torchvision.models import vgg19
from diffusers import DiffusionPipeline, DDIMScheduler, PNDMScheduler

class Vgg19(torch.nn.Module):
    def __init__(self):
        
        super(Vgg19, self).__init__()
        features = list(vgg19(pretrained = True).features)
        self.features = torch.nn.ModuleList(features).eval()
        
    def forward(self, x):
        
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            
            if ii in {1, 6, 11, 20, 29}:
                results.append(x)
        return results

logger.info('We use DDIM Scheduler')
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
            clip_sample=False, set_alpha_to_one=False)
MY_TOKEN = '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'
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 5.
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
ldm = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256", scheduler=scheduler).to(device)
PNDM_scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
# ldm = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256", scheduler=PNDM_scheduler).to(device)
ldm.scheduler.set_timesteps(NUM_DDIM_STEPS)
try:
    ldm.disable_xformers_memory_efficient_attention()
except AttributeError:
    print("Attribute disable_xformers_memory_efficient_attention() is missing")
tokenizer = ldm.tokenizer
#--Additional 
# vgg_features = Vgg19().to(device)
