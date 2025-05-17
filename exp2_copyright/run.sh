#!/bin/bash
set -e

echo "Step 1: Download and preprocess dataset"
python utils/download_data.py
python utils/preprocess.py

echo "Step 2: Pre-train the model"
python pretrain.py --brand_filter pretrain

echo "Step 3: Fine-tune the model for each group"
python train_srs.py --brand_filter grouped

for brand in GoogleSmall GoogleLarge Google SpriteSmall SpriteLarge Sprite Starbucks Vodafone; do
    python train_fsrs.py --fsrs_s0_brand "$brand"
done

echo "Step 4: Generate images"
for brand in Google Sprite Starbucks Vodafone; do
    prompt="A logo by ${brand}"
    save_dir="./save/generated/${brand,,}"  # lowercase
    python inference.py \
        --inference_weights_path ./save/weights_full.pth \
        --prompt "$prompt" \
        --inference_save_dir "$save_dir"
done

echo "Step 5: Compute utility (SRS and FSRS)"
for brand in Google Sprite Starbucks Vodafone; do
    brand_lc=${brand,,}
    x_gen_path="./save/generated/${brand_lc}/a_logo_by_${brand_lc}_latents.pkl"
    save_path="./save/srs/utility/log_likelihoods_${brand_lc}.pkl"

    python utility_srs.py \
        --prompt "A logo by ${brand}" \
        --utility_x_gen_path "$x_gen_path" \
        --srs_utility_save_path "$save_path"

    python utility_fsrs_linear.py --fsrs_x_gen_brand "$brand"

    for s0 in Google GoogleLarge GoogleSmall Sprite SpriteLarge SpriteSmall Starbucks Vodafone; do
        python utility_fsrs_correction.py --fsrs_x_gen_brand "$brand" --fsrs_s0_brand "$s0"
    done
done

echo "Step 6: Aggregate and plot results"
python main.py

echo "All steps completed successfully."