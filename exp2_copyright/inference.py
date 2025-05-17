import os
import pickle
import torch

from peft import set_peft_model_state_dict
from PIL import Image
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

device = config.device
vae, tokenizer, text_encoder, scheduler, unet = get_model()
unet.eval()

lora_weights = torch.load(config.inference_weights_path, map_location=device)
set_peft_model_state_dict(unet, lora_weights)

height, width = config.height, config.width
num_inference_steps = config.num_steps
guidance_scale = config.guidance_scale
num_images = config.inference_num_images
generator = torch.Generator(device=device)

output_dir = config.inference_save_dir
os.makedirs(output_dir, exist_ok=True)

formatted_prompt = config.prompt.lower().replace(" ", "_")

prompt_batch = [config.prompt] * num_images 

text_input = tokenizer(
    prompt_batch, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    ).to(device)

with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids)[0]

max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer([""] * num_images, padding="max_length", max_length=max_length, return_tensors="pt").to(device)
uncond_embeddings = text_encoder(uncond_input.input_ids)[0]
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

latents = torch.randn(
    (num_images, unet.config.in_channels, height // 8, width // 8),
    generator=generator,
    device=device
) * scheduler.init_noise_sigma

scheduler.set_timesteps(num_inference_steps)

for t in tqdm(scheduler.timesteps, miniters=5):
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    latents = scheduler.step(noise_pred, t, latents).prev_sample

latents = latents / 0.18215

latent_save_path = os.path.join(output_dir, f"{formatted_prompt}_latents.pkl")
with open(latent_save_path, "wb") as f:
    pickle.dump(latents.detach().cpu(), f)
print(f"Saved latents to {latent_save_path}")

with torch.no_grad():
    image = vae.decode(latents).sample

image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(img) for img in images]

for idx, img in enumerate(pil_images, start=1):
    image_path = os.path.join(output_dir, f"{formatted_prompt}_{idx}.png")
    img.save(image_path)

print(f"Generated {num_images} images saved in {output_dir}")


