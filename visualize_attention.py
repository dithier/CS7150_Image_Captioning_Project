"""
Authors: Priyanshu Ranka,  Carter Ithier
Course: CS 7150 - Deep Learning
Semester: Spring 2026
Short description:  Script to visualize attention maps for the ResNet + Transformer decoder model, in the style of the SAT paper Figure 5. 
"""

"""
visualize_attention_sat.py

SAT paper Figure 5 style attention visualization for the ResNet + Transformer decoder model.
ResNet50 (no avgpool) → (B, 49, embed_dim) 7x7 spatial grid → cross-attended by decoder.

Controls:
  → (right arrow) : next image in directory
  ← (left arrow)  : previous image
  S               : save current figure to --output_dir
  Q / Escape      : quit

Usage:
    python visualize_attention_sat.py \
        --checkpoint path/to/checkpoint.pth \
        --vocab_path path/to/vocab.pkl \
        --image_dir flickr8k/images \
        --output_dir attention_outputs
"""

import argparse
import math
import os
import pickle

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Patch decoder to expose cross-attention weights
# ──────────────────────────────────────────────────────────────────────────────

def patch_decoder_for_attn_weights(transformer_decoder):
    """
    Forces need_weights=True on each TransformerDecoderLayer's cross-attention
    and caches result on layer._cached_attn_weights after each forward pass.
    """
    for layer in transformer_decoder.layers:
        def make_new_forward(layer):
            def new_forward(tgt, memory, tgt_mask=None, memory_mask=None,
                            tgt_key_padding_mask=None, memory_key_padding_mask=None,
                            tgt_is_causal=False, memory_is_causal=False):
                x = tgt
                if layer.norm_first:
                    x = x + layer._sa_block(layer.norm1(x), tgt_mask,
                                            tgt_key_padding_mask, tgt_is_causal)
                    normed = layer.norm2(x)
                    ca_out, attn_w = layer.multihead_attn(
                        normed, memory, memory,
                        attn_mask=memory_mask,
                        key_padding_mask=memory_key_padding_mask,
                        need_weights=True,
                        average_attn_weights=True,
                    )
                    ca_out = layer.dropout(ca_out)
                    x = x + ca_out
                    x = x + layer._ff_block(layer.norm3(x))
                else:
                    x = layer.norm1(x + layer._sa_block(x, tgt_mask,
                                                        tgt_key_padding_mask, tgt_is_causal))
                    ca_out, attn_w = layer.multihead_attn(
                        x, memory, memory,
                        attn_mask=memory_mask,
                        key_padding_mask=memory_key_padding_mask,
                        need_weights=True,
                        average_attn_weights=True,
                    )
                    ca_out = layer.dropout(ca_out)
                    x = layer.norm2(x + ca_out)
                    x = layer.norm3(x + layer._ff_block(x))

                layer._cached_attn_weights = attn_w.detach().cpu()  # (B, Lq, 49)
                return x
            return new_forward
        layer.forward = make_new_forward(layer)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Generate caption + per-word attention maps
# ──────────────────────────────────────────────────────────────────────────────

def generate_with_attention(model, image):
    """
    Returns:
        words         : list of str
        attention_maps: list of (7, 7) numpy arrays, one per word
    """
    model.eval()
    assert image.size(0) == 1

    # patch the decoder (model.decoder.transformer_decoder for ResNet model)
    patch_decoder_for_attn_weights(model.decoder.transformer_decoder)

    vocab   = model.vocab
    sos_idx = vocab.word_to_index[vocab.SOS_TOKEN]
    eos_idx = vocab.word_to_index[vocab.EOS_TOKEN]

    generated = torch.full((1, 1), sos_idx, dtype=torch.long, device=image.device)

    with torch.no_grad():
        encoder_output = model.encoder(image)  # (1, 49, embed_dim)

    words          = []
    attention_maps = []

    with torch.no_grad():
        for _ in range(30):
            decoder_input = model.decoder.positional_encoding(
                model.decoder.embedding(generated) * math.sqrt(model.decoder.embed_dim)
            )
            dec_out    = model.decoder.transformer_decoder(decoder_input, encoder_output)
            logits     = model.decoder.fc_out(dec_out)
            next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated  = torch.cat((generated, next_token), dim=1)

            idx = next_token[0].item()
            if idx == eos_idx:
                break

            word = vocab.index_to_word[idx]
            words.append(word)

            # attn_w: (1, Lq, 49) — take last query position → (49,) → 7x7
            last_layer = list(model.decoder.transformer_decoder.layers)[-1]
            attn_w     = last_layer._cached_attn_weights
            attn       = attn_w[0, -1, :].numpy()
            attn       = attn.reshape(7, 7)
            attention_maps.append(attn)

    return words, attention_maps


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Pick key word (most focused attention, skip stop words)
# ──────────────────────────────────────────────────────────────────────────────

STOP = {"a", "an", "the", "is", "are", "was", "were", "in", "on",
        "of", "with", "and", "at", "to", "its", "it", "this",
        "that", "there", "their", "has", "have", "by", "for"}

def pick_key_word(words, attention_maps):
    best_idx, best_score = 0, -1.0
    for i, (word, attn) in enumerate(zip(words, attention_maps)):
        if word.lower() in STOP:
            continue
        score = float(attn.max())
        if score > best_score:
            best_score = score
            best_idx   = i
    return best_idx, words[best_idx], attention_maps[best_idx]


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Make spotlight overlay (dark base, 80% white on attended region)
# ──────────────────────────────────────────────────────────────────────────────

def make_spotlight(image_pil, attn_7x7):
    W, H = image_pil.size

    # dark grayscale base
    gray     = np.array(image_pil.convert("L")).astype(float) / 255.0
    gray     = gray * 0.35
    gray_rgb = np.stack([gray, gray, gray], axis=-1)

    # upsample 7x7 attention to image size
    attn_img = Image.fromarray((attn_7x7 * 255).astype(np.uint8)).resize(
        (W, H), resample=Image.BILINEAR
    )
    attn_np = np.array(attn_img).astype(float) / 255.0
    attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)
    attn_np = attn_np ** 0.5   # gamma sharpen

    # 80% white opacity at peak
    alpha  = attn_np[:, :, np.newaxis] * 0.80
    result = gray_rgb * (1 - alpha) + np.ones_like(gray_rgb) * alpha
    result = (result * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(result)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Draw figure
# ──────────────────────────────────────────────────────────────────────────────

def draw_figure(fig, image_pil, words, attn_maps, img_name):
    fig.clf()
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(wspace=0.05, bottom=0.15)

    key_idx, key_word, key_attn = pick_key_word(words, attn_maps)
    spotlight = make_spotlight(image_pil, key_attn)

    axes[0].imshow(np.array(image_pil.convert("RGB")))
    axes[0].axis("off")

    axes[1].imshow(np.array(spotlight))
    axes[1].axis("off")

    # Render caption with key word in uppercase
    parts = [w.upper() if i == key_idx else w for i, w in enumerate(words)]
    fig.text(0.5, 0.04, " ".join(parts) + ".", ha="center", va="top", fontsize=10)
    fig.suptitle(img_name, fontsize=8, color="gray", y=0.98)

    print(f"[{img_name}]  →  {' '.join(words)}  (key: {key_word})")
    print("  S=save   →=next   ←=prev   Q/Esc=quit")
    fig.canvas.draw()


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Interactive loop
# ──────────────────────────────────────────────────────────────────────────────

def run_interactive(model, image_dir, transform, output_dir, all_images=None):
    if all_images is None:
        all_images = sorted([
            f for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
        ])
    if not all_images:
        print(f"No images found in {image_dir}")
        return

    print(f"Browsing {len(all_images)} images from {image_dir}")

    cache = {}
    state = {"idx": 0}
    fig   = plt.figure(figsize=(8, 4))

    def get_result(idx):
        img_name = all_images[idx]
        if img_name not in cache:
            img_path         = os.path.join(image_dir, img_name)
            image_pil        = Image.open(img_path).convert("RGB")
            image_tensor     = transform(image_pil).unsqueeze(0).to(device)
            words, attn_maps = generate_with_attention(model, image_tensor)
            cache[img_name]  = (image_pil, words, attn_maps)
        return img_name, *cache[all_images[idx]]

    def show_current():
        img_name, image_pil, words, attn_maps = get_result(state["idx"])
        draw_figure(fig, image_pil, words, attn_maps, img_name)

    def on_key(event):
        if event.key == "right":
            state["idx"] = (state["idx"] + 1) % len(all_images)
            show_current()
        elif event.key == "left":
            state["idx"] = (state["idx"] - 1) % len(all_images)
            show_current()
        elif event.key in ("s", "S"):
            img_name  = all_images[state["idx"]]
            save_name = os.path.splitext(img_name)[0] + "_attention.png"
            save_path = os.path.join(output_dir, save_name)
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"  Saved → {save_path}")
        elif event.key in ("q", "escape"):
            plt.close("all")

    fig.canvas.mpl_connect("key_press_event", on_key)
    show_current()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/resnet_transformer_best.pt")
    parser.add_argument("--vocab_path", default="vocab.pkl")
    parser.add_argument("--image_dir",         default="flickr8k/images",
                        help="Path to images folder")
    parser.add_argument("--dataset_dir",       default="flickr8k",
                        help="Root flickr8k dir (for building test split)")
    parser.add_argument("--output_dir",        default="attention_outputs")
    # model hyperparams — must match checkpoint
    parser.add_argument("--embed_dim",         type=int, default=256)
    parser.add_argument("--num_heads",         type=int, default=8)
    parser.add_argument("--ff_dim",            type=int, default=1024)
    parser.add_argument("--num_decoder_cells", type=int, default=3)
    parser.add_argument("--dropout",           type=float, default=0.3)
    parser.add_argument("--freeze",            type=bool, default=True)
    args = parser.parse_args()

    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)

    from resnet_transformer_decoder.resnet_transformer import ResnetTransformerModel
    model = ResnetTransformerModel(
        vocab=vocab,
        num_heads=args.num_heads,
        trx_ff_dim=args.ff_dim,
        num_decoder_cells=args.num_decoder_cells,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        freeze=args.freeze,
    ).to(device)

    ckpt  = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    model.eval()
    print("Model loaded.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    # build test split using same seed/split as training
    from dataloader_v2 import get_flickr8k_loaders
    _, _, test_loader, _ = get_flickr8k_loaders(root_dir=args.dataset_dir)
    seen, test_images = set(), []
    for name in test_loader.dataset.images:
        if name not in seen:
            seen.add(name)
            test_images.append(name)
    print(f"Test split: {len(test_images)} images")

    run_interactive(model, args.image_dir, transform, args.output_dir,
                    all_images=test_images)


if __name__ == "__main__":
    main()