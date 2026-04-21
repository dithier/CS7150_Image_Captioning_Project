"""
Authors: Carter Ithier, Priyanshu Ranka  
Course: CS 7150 - Deep Learning
Semester: Spring 2026
Short description:  Script to download pre-trained image captioning models. 
"""

import os
import urllib.request

MODEL_DIR = "models"


def download_model(model_url):
    os.makedirs(MODEL_DIR, exist_ok=True)

    filename = model_url.split("/")[-1]
    dest = os.path.join(MODEL_DIR, filename)

    if os.path.exists(dest):
        print(f"Model already exists at {dest}, skipping download.")
        return dest

    print(f"Downloading {filename}...")

    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded / total_size * 100, 100)
        mb_done = downloaded / 1e6
        mb_total = total_size / 1e6
        print(f"\r  {percent:.1f}%  ({mb_done:.1f} / {mb_total:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(model_url, dest, reporthook=show_progress)
    print(f"\nSaved to {dest}")
    return dest


if __name__ == "__main__":
    resnet_transformer_model = "https://github.com/dithier/CS7150_Image_Captioning_Project/releases/download/v1/resnet_transformer_best.pt"

    baseline_model = "https://github.com/dithier/CS7150_Image_Captioning_Project/releases/download/v2/baseline_best_model.pt"

    vit_decoder_model = "https://github.com/dithier/CS7150_Image_Captioning_Project/releases/download/v3/vit_best_model.pt"

    model_urls = [resnet_transformer_model, baseline_model, vit_decoder_model]

    for model_url in model_urls:
        if model_url:
            download_model(model_url)

