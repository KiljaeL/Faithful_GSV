import torch
import argparse

def get_config():
    parser = argparse.ArgumentParser(description="Experiment Configuration")

    parser.add_argument("--raw_image_dir", type=str, default="./data/flickr_logos_27_dataset_images",
                        help="Path to the directory containing raw images.")
    parser.add_argument("--training_annotation", type=str, default="./data/flickr_logos_27_dataset_training_set_annotation.txt",
                        help="Path to training annotation file.")
    
    parser.add_argument("--processed_image_dir", type=str, default="./data/flickr_logos_27_dataset_processed/images",
                        help="Path to the directory for storing processed images.")
    parser.add_argument("--processed_prompt_dir", type=str, default="./data/flickr_logos_27_dataset_processed/prompts",
                        help="Path to the directory for storing processed prompts.")
    parser.add_argument("--output_folder", type=str, default="./data/flickr_logos_27_dataset_processed/pickles",
                        help="Path to the directory for storing pickle files.")


    parser.add_argument("--height", type=int, default=256, help="Height of the generated image.")
    parser.add_argument("--width", type=int, default=256, help="Width of the generated image.")
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=device, help="Device to use for training (cuda/mps/cpu).")
    parser.add_argument("--dtype", type=str, default="float16" if device == "cuda" else "float32",
                        help="Data type precision: float16 for GPU, float32 otherwise.")
    
    parser.add_argument("--data_dir", type=str, default="./data/flickr_logos_27_dataset_processed/pickles",
                    help="Path to the dataset directory.")
    
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for data loading.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")

    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank (dimension).")
    parser.add_argument("--lora_alpha", type=int, default=8, help="LoRA scaling factor.")
    
    brand_filter_all = [
        'Adidas', 'Apple', 'BMW', 'Citroen', 'Cocacola', 'DHL', 'Fedex', 'Ferrari',
        'Ford', 'Google', 'Heineken', 'HP', 'Intel', 'McDonalds', 'Mini', 'Nbc', 'Nike',
        'Pepsi', 'Porsche', 'Puma', 'RedBull', 'Sprite', 'Starbucks', 'Texaco', 'Unicef',
        'Vodafone', 'Yahoo'
    ]
    brand_filter_ungrouped = ['Google', 'Sprite', 'Starbucks', 'Vodafone']
    brand_filter_grouped = ['GoogleLarge', 'GoogleSmall', 'SpriteLarge', 'SpriteSmall', 'Starbucks', 'Vodafone']
    brand_filter_pretrain = [b for b in brand_filter_all if b not in brand_filter_ungrouped]
    
    parser.add_argument("--brand_filter", type=str, choices=["pretrain", "ungrouped", "grouped"], default="grouped",
                        help="Choose the brand filtering method: pretrain, ungrouped, or grouped.")

    parser.add_argument("--inference_weights_path", type=str, default=None, help="Path to the fine-tuned LoRA weights (.pth file)")
    parser.add_argument("--inference_num_images", type=int, default=20, help="Total number of images to generate")
    parser.add_argument("--inference_save_dir", type=str, default="./save/generated", help="Path to save the generated image")
    
    parser.add_argument("--num_steps", type=int, default=25, help="Number of inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for generation.")
    parser.add_argument("--prompt", type=str, default="A logo by Google", help="Text prompt for generation")
    
    parser.add_argument("--pretrained_weights_path", type=str, default="./save/weights_null.pth", help="Path to the pretrained weights file.")
    parser.add_argument("--finetuned_weights_path", type=str, default="./save/weights_full.pth", help="Path to the fully-finetuned weights file.")
    parser.add_argument("--utility_x_gen_path", type=str, default="./save/generated/google/a_logo_by_google_latents.pkl", help="Path to the x_gen file.")
    parser.add_argument("--srs_weights_dir", type=str, default="./save/srs/weight", help="Directory to save fine-tuned weights.")
    parser.add_argument("--srs_utility_save_path", type=str, default="./save/srs/utility/log_likelihoods.pkl", help="Path to save the log-likelihoods.") 
    parser.add_argument("--utility_num_samples", type=int, default=20, help="Number of samples for Monte Carlo estimation.")


    parser.add_argument("--fsrs_weights_dir", type=str, default="./save/fsrs/weight", help="Directory where the finetuned weights are saved.")
    parser.add_argument("--fsrs_x_gen_dir", type=str, default="./save/generated", help="Directory where the x_gen files are saved.")
    parser.add_argument("--fsrs_utility_save_dir", type=str, default="./save/fsrs/utility", help="Directory where the log-likelihoods are saved.")
    parser.add_argument("--fsrs_x_gen_brand", type=str, default="Google", help="Brand for GShap x_gen.")
    parser.add_argument("--fsrs_s0_brand", type=str, default="Google", help="Brand for GShap S0.")
    parser.add_argument("--fsrs_h", type=int, default=1, help="Step for finite difference.")
    parser.add_argument("--fsrs_m", type=int, default=2, help="Number of Monte Carlo samples.")
   
    args = parser.parse_args()
    
    if args.brand_filter == "pretrain":
        args.brand_filter = brand_filter_pretrain
    elif args.brand_filter == "ungrouped":
        args.brand_filter = brand_filter_ungrouped
    elif args.brand_filter == "grouped":
        args.brand_filter = brand_filter_grouped

    
    return args

config = get_config()