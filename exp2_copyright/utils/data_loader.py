import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import torch
import random
from torch.utils.data import Dataset, DataLoader
from config import config

class LoadDataset(Dataset):
    def __init__(self, data_dir, brand_filter=None, fsrs_config=None):
        self.data_dir = data_dir
        self.brand_filter = brand_filter if brand_filter is not None else [] 
        self.pkl_files = sorted(os.listdir(data_dir))
        self.fsrs_config = fsrs_config

        if fsrs_config:
            s0_brand = fsrs_config["s0_brand"]
            s0_files = [f for f in self.pkl_files if f.startswith(s0_brand)]
            other_brands = [b for b in self.brand_filter if not b.startswith(s0_brand)]
            other_files = [f for f in self.pkl_files if any(f.startswith(b) for b in other_brands)]
            
            n = 120
            s = fsrs_config["s"]
            s0 = len(s0_files)
            s1_min = max(0, s0 + s - n)
            s1_max = min(s0, s)
            Es1 = round(s0 * s / n)
            h = fsrs_config["h"]
            print(f"\ns: {s}, s0: {s0}, n: {n}, Es1: {Es1}, h: {h}")
            if Es1 + h <= s1_max and Es1 - h >= s1_min:  # Central difference
                common_s0 = random.sample(s0_files, Es1 - 1)
                common_other = random.sample(other_files, s - Es1 - 1)
                
                self.pkl_files_common = common_s0 + common_other
                self.pkl_files_lower = random.sample(list(set(other_files) - set(common_other)), 2)
                self.pkl_files_upper = random.sample(list(set(s0_files) - set(common_s0)), 2)

            elif Es1 + h <= s1_max:  # Forward difference
                common_s0 = random.sample(s0_files, Es1)
                common_other = random.sample(other_files, s - Es1 - 1)
                
                self.pkl_files_common = common_s0 + common_other
                self.pkl_files_lower = random.sample(list(set(other_files) - set(common_other)), 1)
                self.pkl_files_upper = random.sample(list(set(s0_files) - set(common_s0)), 1)

            elif Es1 - h >= s1_min:  # Backward difference
                common_s0 = random.sample(s0_files, Es1 - 1)
                common_other = random.sample(other_files, s - Es1)
                
                self.pkl_files_common = common_s0 + common_other
                self.pkl_files_lower = random.sample(list(set(other_files) - set(common_other)), 1)
                self.pkl_files_upper = random.sample(list(set(s0_files) - set(common_s0)), 1)
                
            print("Common files:", self.pkl_files_common)
            print("Lower files:", self.pkl_files_lower)
            print("Upper files:", self.pkl_files_upper)
            
        elif self.brand_filter:
            self.pkl_files = [
                pkl_file for pkl_file in self.pkl_files 
                if any(brand in pkl_file for brand in self.brand_filter)
            ]
            print("Filtered files:", self.pkl_files)

    def __len__(self):
        return len(self.pkl_files)

class SimpleDataset(Dataset):
    def __init__(self, images, prompts):
        self.images = images
        self.prompts = prompts

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.prompts[idx]

def load_samples(data_dir, file_list):
    images = []
    prompts = []
    for fname in file_list:
        path = os.path.join(data_dir, fname)
        with open(path, "rb") as f:
            data = pickle.load(f)
        image = torch.tensor(data["image"]).squeeze(0)
        prompt = data["prompt"]
        images.append(image)
        prompts.append(prompt)
    return images, prompts

def get_data(user_brand_filter=None, fsrs_config=None):
    if user_brand_filter:
        config.brand_filter = user_brand_filter

    if fsrs_config:
        dataset = LoadDataset(config.data_dir, config.brand_filter, fsrs_config)
        
        images_common, prompts_common = load_samples(config.data_dir, dataset.pkl_files_common)
        dataset_common = SimpleDataset(images_common, prompts_common)
        if len(dataset_common) == 0:
            dataloader_common = []
        else:
            dataloader_common = DataLoader(dataset_common, batch_size=config.batch_size, shuffle=True, num_workers=4)
        
        images_lower, prompts_lower = load_samples(config.data_dir, dataset.pkl_files_lower)
        dataset_lower = SimpleDataset(images_lower, prompts_lower)
        dataloader_lower = DataLoader(dataset_lower, batch_size=config.batch_size, shuffle=True, num_workers=4)
        
        images_upper, prompts_upper = load_samples(config.data_dir, dataset.pkl_files_upper)
        dataset_upper = SimpleDataset(images_upper, prompts_upper)
        dataloader_upper = DataLoader(dataset_upper, batch_size=config.batch_size, shuffle=True, num_workers=4)

        print(f"Loaded {len(dataset_common)} + {len(dataset_lower)} + {len(dataset_upper)} samples")

        return dataloader_common, dataloader_lower, dataloader_upper

    
    else:
        dataset = LoadDataset(config.data_dir, config.brand_filter)
        images, prompts = load_samples(config.data_dir, dataset.pkl_files)
        dataset = SimpleDataset(images, prompts)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

        print(f"Loaded {len(dataset)} samples from '{config.data_dir}'")
        return dataloader

if __name__ == "__main__":
    train_dataloader = get_data()
    print(f"Batch size: {next(iter(train_dataloader))[0].shape}")