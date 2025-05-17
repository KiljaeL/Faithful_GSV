import os

import torch
import torch.nn.functional as F
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

# Check if CUDA is available and enable FP16 accordingly
device = config.device  # Automatic device selection
if device == "cuda":
    print(f"CUDA is enabled. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Running on CPU.")

def save_lora_weights(unet, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(get_peft_model_state_dict(unet), save_path)
    print(f"LoRA weights saved to {save_path}")

def fsrs_train_loop():
    dtype = torch.float32
    device = config.device
    n = 120
    m = config.fsrs_m
    h = config.fsrs_h
    
    vae, tokenizer, text_encoder, scheduler, unet = get_model()
    
    weights_pretrained = torch.load(config.pretrained_weights_path, map_location=device)
    fsrs_config = {"s0_brand": config.fsrs_s0_brand, "h": h}
    
    unet.train()
    
    for s in tqdm(range(1, n), desc="Subset size"):
        for iter in range(m):
            fsrs_config["s"] = s
            dataloader_common, dataloader_lower, dataloader_upper = get_data(fsrs_config=fsrs_config)
            
            set_peft_model_state_dict(unet, weights_pretrained)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, unet.parameters()),
                lr=config.learning_rate,
                eps=1e-6
            )
            
            if s > 1:
                for epoch in range(config.num_epochs):
                    total_loss = 0.0
                    for step, (image, prompt) in tqdm(enumerate(dataloader_common), total=len(dataloader_common), disable=True):
                        image = image.to(config.device, dtype=dtype)
                        latent = vae.encode(image).latent_dist.sample() * 0.18215

                        input_ids = tokenizer(prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(config.device)
                        encoder_hidden_states = text_encoder(input_ids, return_dict=False)[0]

                        timesteps = torch.randint(0, 1000, (latent.shape[0],), device=config.device, dtype=torch.long)  
                        
                        noise = torch.randn_like(latent, dtype=dtype)

                        noisy_latent = scheduler.add_noise(latent, noise, timesteps)
                        
                        model_pred = unet(noisy_latent, timesteps, encoder_hidden_states, return_dict=False)[0]

                        loss = F.mse_loss(model_pred, noise, reduction="mean")

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()

                    print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {total_loss / len(dataloader_common)}")
                
            weights_common = get_peft_model_state_dict(unet).copy()
    
            set_peft_model_state_dict(unet, weights_common)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, unet.parameters()),
                lr=config.learning_rate,
                eps=1e-6
            )
            
            for epoch in range(config.num_epochs):
                total_loss = 0.0
                for step, (image, prompt) in tqdm(enumerate(dataloader_lower), total=len(dataloader_lower), disable=True):
                    image = image.to(config.device, dtype=dtype)
                    latent = vae.encode(image).latent_dist.sample() * 0.18215

                    input_ids = tokenizer(prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(config.device)
                    encoder_hidden_states = text_encoder(input_ids, return_dict=False)[0]

                    timesteps = torch.randint(0, 1000, (latent.shape[0],), device=config.device, dtype=torch.long)      

                    noise = torch.randn_like(latent, dtype=dtype)

                    noisy_latent = scheduler.add_noise(latent, noise, timesteps)    
                    
                    model_pred = unet(noisy_latent, timesteps, encoder_hidden_states, return_dict=False)[0]

                    loss = F.mse_loss(model_pred, noise, reduction="mean")

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()    
                    
                    total_loss += loss.item()

                print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {total_loss / len(dataloader_lower)}")   
            
            save_lora_weights(unet, save_path=config.fsrs_weights_dir + f"/{config.fsrs_s0_brand.lower()}/weights_lower_{s}_{iter+1}.pth")
                
            set_peft_model_state_dict(unet, weights_common)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, unet.parameters()),
                lr=config.learning_rate,
                eps=1e-6
            )   
            
            for epoch in range(config.num_epochs):
                total_loss = 0.0
                for step, (image, prompt) in tqdm(enumerate(dataloader_upper), total=len(dataloader_upper), disable=True):
                    image = image.to(config.device, dtype=dtype)
                    latent = vae.encode(image).latent_dist.sample() * 0.18215
                    
                    input_ids = tokenizer(prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(config.device)
                    encoder_hidden_states = text_encoder(input_ids, return_dict=False)[0]

                    timesteps = torch.randint(0, 1000, (latent.shape[0],), device=config.device, dtype=torch.long)

                    noise = torch.randn_like(latent, dtype=dtype)

                    noisy_latent = scheduler.add_noise(latent, noise, timesteps)

                    model_pred = unet(noisy_latent, timesteps, encoder_hidden_states, return_dict=False)[0]

                    loss = F.mse_loss(model_pred, noise, reduction="mean")

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()    
                    
                    total_loss += loss.item()

                print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {total_loss / len(dataloader_upper)}")
                
            save_lora_weights(unet, save_path=config.fsrs_weights_dir + f"/{config.fsrs_s0_brand.lower()}/weights_upper_{s}_{iter+1}.pth")
            
if __name__ == "__main__":
    print("Starting training...")
    fsrs_train_loop()
    print("Training completed!")