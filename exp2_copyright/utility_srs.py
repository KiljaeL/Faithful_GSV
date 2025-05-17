import os

import pickle
import torch
import torch.nn.functional as F
from itertools import chain, combinations
from peft import set_peft_model_state_dict
from tqdm.auto import tqdm
from config import config
from utils.model_setup import get_model
import numpy as np
import random

seed = 1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def powerset(iterable):
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(0, len(s) + 1)))

def simplify_subsets(subsets):
    simplified = []
    for subset in subsets:
        brands = set()
        for item in subset:
            if 'Large' in item or 'Small' in item:
                brand = item.split('Large')[0]
                brand = brand.split('Small')[0]
            else:
                brand = item
            brands.add(brand)
        
        new_subset = []
        for brand in brands:
            if f"{brand}Large" in subset and f"{brand}Small" in subset:
                new_subset.append(brand)
            else:
                if f"{brand}Large" in subset:
                    new_subset.append(f"{brand}Large")
                elif f"{brand}Small" in subset:
                    new_subset.append(f"{brand}Small")
                else:
                    new_subset.append(brand)
        
        simplified.append(tuple(sorted(new_subset)))
    
    return sorted(list(set(simplified)), key=lambda x: (len(x), x))

def srs_compute_utility():
    device = config.device
    
    _, tokenizer, text_encoder, scheduler, unet = get_model()
    unet.eval()
    
    height, width = config.height, config.width
    num_steps = config.num_steps
    guidance_scale = config.guidance_scale
    num_samples = config.utility_num_samples
    generator = torch.Generator(device=device).manual_seed(seed)
    
    x_gen_path = config.utility_x_gen_path
    
    with open(x_gen_path, "rb") as f:
        x_gen = pickle.load(f)
        
    x_gen = torch.tensor(x_gen, device=device) if not isinstance(x_gen, torch.Tensor) else x_gen.clone().detach().to(device)
    num_gen = len(x_gen)
    
    prompt_batch = [config.prompt] * num_samples 

    text_input = tokenizer(
        prompt_batch, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        ).to(device)

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids)[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * num_samples, padding="max_length", max_length=max_length, return_tensors="pt").to(device)
    uncond_embeddings = text_encoder(uncond_input.input_ids)[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    brand_filter = config.brand_filter
    subsets = simplify_subsets(powerset(brand_filter))
    
    log_likelihoods_all = dict()
    
    for subset in subsets:
        subset = list(subset)
        
        print(f"Computing utility on subset: {subset}")
        
        if len(subset) == 0:
            load_path = config.srs_weights_dir + "/weights_null.pth"
        elif subset == ["Google", "Sprite", "Starbucks", "Vodafone"]:
            load_path = config.srs_weights_dir + "/weights_full.pth"
        else:
            load_path = config.srs_weights_dir + "/weights_" + "_".join(subset) + ".pth"
        
        print(f"Loading model from {load_path}")
        
        lora_weights = torch.load(load_path, map_location=device)
        set_peft_model_state_dict(unet, lora_weights)
        
        generator = torch.Generator(device=device).manual_seed(seed)
    
        latents = torch.randn(
            (num_samples, unet.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=device
        ) * scheduler.init_noise_sigma

        scheduler.set_timesteps(num_steps)

        for t in tqdm(scheduler.timesteps, miniters=5):

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

        log_likelihoods = []
    
    
        for i in range(num_gen):
            x_gen_i = x_gen[i].unsqueeze(0).expand(num_samples, -1, -1, -1)
            
            mse = F.mse_loss(x_gen_i, latents, reduction="mean")
            
            log_likelihoods.append(float(-mse))
        
        if len(subset) == 0:
            log_likelihoods_all["null"] = log_likelihoods
        elif subset == ["Google", "Sprite", "Starbucks", "Vodafone"]:
            log_likelihoods_all["full"] = log_likelihoods
        else:
            log_likelihoods_all["_".join(subset)] = log_likelihoods
    
    save_path = config.srs_utility_save_path
    with open(save_path, "wb") as f:
        pickle.dump(log_likelihoods_all, f)
    print(f"Log-likelihoods saved to {save_path}")
    
if __name__ == "__main__":
    print("Starting utility computation...")
    srs_compute_utility()
    print("Utility computation completed.")