import os
import urllib.request

# current is for resnet
MODEL_URL = "https://github.com/dithier/CS7150_Image_Captioning_Project/releases/download/v1/resnet_transformer_best.pt"
MODEL_DIR = "models"


def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    filename = MODEL_URL.split("/")[-1]
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

    urllib.request.urlretrieve(MODEL_URL, dest, reporthook=show_progress)
    print(f"\nSaved to {dest}")
    return dest


if __name__ == "__main__":
    download_model()

