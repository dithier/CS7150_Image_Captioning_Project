"""
Caption a single image using baseline_model_v2 (WITH learned c0 init layer).
Use this for checkpoints trained with baseline_model_v2.py (e.g. SGD v2, adam_baseline_v2).

Usage:
    python caption_v2_with_c.py --image_path path/to/image.jpg --checkpoint_path saved_models/xxx/model.pt
"""
import torch
import argparse
from PIL import Image
from torchvision import transforms
from dataloader_v2 import get_flickr8k_loaders
from baseline_model_v2 import BaselineModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

def main(opt):
    _, _, _, vocab = get_flickr8k_loaders(root_dir=opt.dataset_dir)

    model = BaselineModel(vocab_size=len(vocab)).to(device)
    checkpoint = torch.load(opt.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} | val loss {checkpoint['val_loss']:.4f}")
    print(f"Model : baseline_model_v2 (with learned c0 init)")

    image = load_image(opt.image_path)
    model.eval()
    output = model.generate(image, vocab)
    caption = " ".join(output[0]) if output[0] else "<empty>"

    print(f"\nImage  : {opt.image_path}")
    print(f"Caption: {caption}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path",      type=str, required=True,  help="path to any image")
    parser.add_argument("--checkpoint_path", type=str, required=True,  help="path to model checkpoint")
    parser.add_argument("--dataset_dir",     type=str, default="flickr8k", help="path to flickr8k folder")
    opt = parser.parse_args()
    main(opt)
