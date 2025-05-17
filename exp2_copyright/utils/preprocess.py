import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import shutil
import pickle
import pandas as pd
from collections import defaultdict
from torchvision import transforms
from PIL import Image
from config import config
import random

random.seed(42)

def process_images():
    raw_image_dir = config.raw_image_dir
    processed_image_dir = config.processed_image_dir
    processed_prompt_dir = config.processed_prompt_dir
    output_folder = config.output_folder

    os.makedirs(processed_image_dir, exist_ok=True)
    os.makedirs(processed_prompt_dir, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    df_training = pd.read_csv(config.training_annotation, sep='\s+', header=None).iloc[:, 0:2]
    df_training.columns = ['image_id', 'brand_id']
    label_mapping = dict(zip(df_training["image_id"], df_training["brand_id"]))

    device = config.device
    transform = transforms.Compose([
        transforms.Resize(config.height, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(config.height),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    split_brands = ['Google', 'Sprite', 'Starbucks', 'Vodafone']
    
    brand_counter = defaultdict(int)
    brand_images = defaultdict(list)

    for img_filename in os.listdir(raw_image_dir):
        if not img_filename.endswith(".jpg"):
            continue

        brand = label_mapping.get(img_filename, "Unknown")
        brand_images[brand].append(img_filename)

    for brand, images in brand_images.items():
        if brand in split_brands:
            random.shuffle(images)
            large_images = images[:20]
            small_images = images[20:]

            for img_filename in large_images:
                process_single_image(img_filename, f"{brand}Large", brand_counter, 
                                  raw_image_dir, processed_image_dir, processed_prompt_dir, 
                                  output_folder, label_mapping, transform, device)

            for img_filename in small_images:
                process_single_image(img_filename, f"{brand}Small", brand_counter, 
                                  raw_image_dir, processed_image_dir, processed_prompt_dir, 
                                  output_folder, label_mapping, transform, device)
        else:
            for img_filename in images:
                process_single_image(img_filename, brand, brand_counter, 
                                  raw_image_dir, processed_image_dir, processed_prompt_dir, 
                                  output_folder, label_mapping, transform, device)

    print("All images & prompts have been successfully processed and saved!")

def process_single_image(img_filename, brand_name, brand_counter, raw_image_dir, 
                        processed_image_dir, processed_prompt_dir, output_folder, 
                        label_mapping, transform, device):
    """Process single image"""
    src_path = os.path.join(raw_image_dir, img_filename)
    brand_counter[brand_name] += 1

    new_filename = f"{brand_name}{brand_counter[brand_name]}.jpg"
    dest_path = os.path.join(processed_image_dir, new_filename)

    # shutil.copy(src_path, dest_path)

    base_brand = brand_name.replace("Large", "").replace("Small", "")
    prompt_text = f"A logo by {base_brand}"
    prompt_path = os.path.join(processed_prompt_dir, new_filename.replace(".jpg", ".txt"))
    
    # with open(prompt_path, "w") as f:
    #     f.write(prompt_text)

    image = Image.open(dest_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    sample_data = {"image": image.cpu().numpy(), "prompt": prompt_text}
    sample_filename = os.path.join(output_folder, f"{new_filename.replace('.jpg', '.pkl')}")

    with open(sample_filename, "wb") as f:
        pickle.dump(sample_data, f)

if __name__ == "__main__":
    print("Starting preprocessing...")
    process_images()
    print("Preprocessing completed!")