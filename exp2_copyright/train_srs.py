import os

import torch
import torch.nn.functional as F
from itertools import chain, combinations
from tqdm.auto import tqdm
from config import config 
from utils.data_loader import get_data 
from utils.model_setup import get_model
from peft import get_peft_model_state_dict, set_peft_model_state_dict
import numpy as np
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

def save_lora_weights(unet, save_path):
    """Saves only the LoRA adapter weights."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(get_peft_model_state_dict(unet), save_path)
    print(f"LoRA weights saved to {save_path}")

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

def srs_train_loop():
    dtype = torch.float32  
    
    vae, tokenizer, text_encoder, scheduler, unet = get_model()
    
    weights_pretrained = torch.load(config.pretrained_weights_path, map_location=device)
    brand_filter = config.brand_filter
    subsets = simplify_subsets(powerset(brand_filter))
    
    unet.train()
    
    for subset in subsets:
        subset = list(subset)
        
        print(f"Training on subset: {subset}")
        
        if len(subset) == 0:
            save_lora_weights(unet, save_path=config.srs_weights_dir + "/weights_null.pth")
            continue
    
        train_dataloader = get_data(subset)
        set_peft_model_state_dict(unet, weights_pretrained)
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, unet.parameters()),  
            lr=config.learning_rate,
            eps=1e-6  
        )
        
        for epoch in range(config.num_epochs):
            total_loss = 0.0
            for step, (image, prompt) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), miniters=5):
                
                image = image.to(config.device, dtype=dtype)
                latent = vae.encode(image).latent_dist.sample() * 0.18215

                input_ids = tokenizer(prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(config.device)
                encoder_hidden_states = text_encoder(input_ids, return_dict=False)[0]

                timesteps = torch.randint(0, 1000, (latent.shape[0],), device=config.device, dtype=torch.long)

                noise = torch.randn_like(latent, dtype=dtype)

                noisy_latent = scheduler.add_noise(latent, noise, timesteps)

                model_pred = unet(noisy_latent, timesteps, encoder_hidden_states, return_dict=False)[0]

                target = noise

                loss = F.mse_loss(model_pred, target, reduction="mean")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {total_loss / len(train_dataloader)}")
            
        if subset == ["Google", "Sprite", "Starbucks", "Vodafone"]:
            save_lora_weights(unet, save_path=config.srs_weights_dir + "/weights_full.pth")
            save_lora_weights(unet, save_path="./save/weights_full.pth")
        else:
            save_lora_weights(unet, save_path=config.srs_weights_dir + "/weights_" + "_".join(subset) + ".pth")
    
if __name__ == "__main__":
    print("Starting training...")
    srs_train_loop()
    print("Training completed!")