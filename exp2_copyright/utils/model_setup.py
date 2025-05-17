import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig
from config import config

def get_model():
    device = config.device
    dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="vae",
        torch_dtype=dtype
    ).to(device)

    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")

    text_encoder = CLIPTextModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="text_encoder",
        torch_dtype=dtype
    ).to(device)

    scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="unet",
        revision="main",
        torch_dtype=dtype
    ).to(device)
    
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=config.lora_r,  
        lora_alpha=config.lora_alpha,  
        init_lora_weights="gaussian",  
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],  
    )
    unet.add_adapter(unet_lora_config)

    print("Model components loaded successfully.")
    return vae, tokenizer, text_encoder, scheduler, unet

if __name__ == "__main__":
    models = get_model()
    print("Model setup completed!")