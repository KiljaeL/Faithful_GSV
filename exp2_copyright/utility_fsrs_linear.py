import os

import pickle
import torch
import torch.nn.functional as F
import numpy as np
from peft import set_peft_model_state_dict
from tqdm.auto import tqdm
from config import config
from utils.model_setup import get_model
import random

seed = 1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = config.device
if device == "cuda":
    print(f"CUDA is enabled. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Running on CPU.")


def fsrs_compute_utility_linear():
    device = config.device
    num_samples = config.utility_num_samples
    
    _, tokenizer, text_encoder, scheduler, unet = get_model()
    unet.eval()
    
    height, width = config.height, config.width
    num_steps = config.num_steps
    guidance_scale = config.guidance_scale
    num_samples = config.utility_num_samples  
    generator = torch.Generator(device=device).manual_seed(seed)
    
    weights_null = torch.load(config.pretrained_weights_path, map_location=device)
    weights_full = torch.load(config.finetuned_weights_path, map_location=device)
    
    x_gen_path = config.utility_x_gen_path
    
    with open(x_gen_path, "rb") as f:
        x_gen = pickle.load(f)
        
    x_gen = torch.tensor(x_gen, device=device) if not isinstance(x_gen, torch.Tensor) else x_gen.clone().detach().to(device)
    num_gen = len(x_gen)
    
    prompt_batch = [f"A logo by {config.fsrs_x_gen_brand}"] * num_samples 

    text_input = tokenizer(
        prompt_batch, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        ).to(device)

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids)[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * num_samples, padding="max_length", max_length=max_length, return_tensors="pt").to(device)
    uncond_embeddings = text_encoder(uncond_input.input_ids)[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    log_likelihoods_null = np.zeros(num_gen)
    log_likelihoods_full = np.zeros(num_gen)
    
    set_peft_model_state_dict(unet, weights_null)
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    latents = torch.randn(
        (num_samples, unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        device=device
    ) * scheduler.init_noise_sigma
    
    scheduler.set_timesteps(num_steps) 
    
    for t in tqdm(scheduler.timesteps, disable=True):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
        
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        if t > 1:
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        else:
            t_idx = t.item() if isinstance(t, torch.Tensor) else t
            alpha_1 = scheduler.alphas_cumprod[t_idx]
            beta_1 = 1 - alpha_1
            latents = (1 / torch.sqrt(alpha_1)) * (latents - (beta_1 / torch.sqrt(1 - alpha_1)) * noise_pred)
            
    for i in range(num_gen):
        x_gen_i = x_gen[i].unsqueeze(0).expand(num_samples, -1, -1, -1)
        mse = F.mse_loss(x_gen_i, latents, reduction="mean")
        log_likelihoods_null[i] = float(-mse)
    
    set_peft_model_state_dict(unet, weights_full)
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    latents = torch.randn(
        (num_samples, unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        device=device
    ) * scheduler.init_noise_sigma
    
    scheduler.set_timesteps(num_steps)
    
    for t in tqdm(scheduler.timesteps, disable=True):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
        
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        if t > 1:
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        else:
            t_idx = t.item() if isinstance(t, torch.Tensor) else t
            alpha_1 = scheduler.alphas_cumprod[t_idx]
            beta_1 = 1 - alpha_1
            latents = (1 / torch.sqrt(alpha_1)) * (latents - (beta_1 / torch.sqrt(1 - alpha_1)) * noise_pred)
            
    for i in range(num_gen):
        x_gen_i = x_gen[i].unsqueeze(0).expand(num_samples, -1, -1, -1)
        mse = F.mse_loss(x_gen_i, latents, reduction="mean")
        log_likelihoods_full[i] = float(-mse)

    save_dir = config.fsrs_utility_save_dir
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"log_likelihoods_linear_{config.fsrs_x_gen_brand.lower()}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"null": log_likelihoods_null, "full": log_likelihoods_full}, f)
    print(f"Log-likelihoods saved to {save_path}")
    
if __name__ == "__main__":
    print("Starting utility computation...")
    fsrs_compute_utility_linear()
    print("Utility computation completed.")