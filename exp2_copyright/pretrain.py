import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from config import config
from utils.data_loader import get_data 
from utils.model_setup import get_model
from peft import get_peft_model_state_dict

device = config.device
if device == "cuda":
    print(f"CUDA is enabled. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Running on CPU.")

def save_lora_weights(unet, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(get_peft_model_state_dict(unet), save_path)
    print(f"LoRA weights saved to {save_path}")


def train_loop():
    train_dataloader = get_data()
    vae, tokenizer, text_encoder, scheduler, unet = get_model()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),  
        lr=config.learning_rate,
        eps=1e-6  
    )

    dtype = torch.float32  

    unet.train()
    for epoch in range(config.num_epochs):
        total_loss = 0.0
        for step, (image, prompt) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            
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

    save_lora_weights(unet, "./save/weights_null.pth")

if __name__ == "__main__":
    print("Starting training...")
    train_loop()
    print("Training completed!")
