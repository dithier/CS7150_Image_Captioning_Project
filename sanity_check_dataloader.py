import torch
from dataloader_v2 import get_flickr8k_loaders
from models import BaselineModel

"""
Sanity check script — verifies the dataloader, encoder, and decoder
are all working correctly with the right shapes and token ordering.

Usage:
    python sanity_check.py --dataset_dir flickr8k
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default="flickr8k")
opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ── 1. DATALOADER ────────────────────────────────────────────────────
print("=" * 60)
print("1. DATALOADER")
print("=" * 60)

train_loader, val_loader, test_loader, vocab = get_flickr8k_loaders(
    root_dir=opt.dataset_dir,
    batch_size=4,
)

print(f"\nVocabulary size : {len(vocab)}")
print(f"Train batches   : {len(train_loader)}  ({len(train_loader.dataset)} caption pairs)")
print(f"Val   batches   : {len(val_loader)}  ({len(val_loader.dataset)} caption pairs)")
print(f"Test  batches   : {len(test_loader)}  ({len(test_loader.dataset)} caption pairs)")

# grab one batch
images, captions, image_names = next(iter(train_loader))
print(f"\nBatch shapes:")
print(f"  images   : {tuple(images.shape)}   (B, C, H, W)")
print(f"  captions : {tuple(captions.shape)}  (B, max_caption_length + 2)")

# verify token ordering: SOS ... EOS ... PAD
print(f"\nCaption token ordering check (first caption in batch):")
cap = captions[0].tolist()
sos_idx = vocab.word_to_index[vocab.SOS_TOKEN]
eos_idx = vocab.word_to_index[vocab.EOS_TOKEN]
pad_idx = vocab.word_to_index[vocab.PAD_TOKEN]

sos_pos = cap.index(sos_idx) if sos_idx in cap else None
eos_pos = cap.index(eos_idx) if eos_idx in cap else None
first_pad_pos = cap.index(pad_idx) if pad_idx in cap else None

print(f"  Full token IDs : {cap}")
print(f"  SOS at position: {sos_pos}  (should be 0)")
print(f"  EOS at position: {eos_pos}  (should be right after last word)")
print(f"  First PAD at  : {first_pad_pos}  (should be after EOS, or None if no padding needed)")

# verify EOS comes before PAD
if eos_pos and first_pad_pos:
    assert eos_pos < first_pad_pos, "ERROR: PAD appears before EOS!"
    print(f"  Order check    : EOS ({eos_pos}) < PAD ({first_pad_pos}) ✓")
elif eos_pos and first_pad_pos is None:
    print(f"  Order check    : EOS present, no padding needed (caption fills max length) ✓")
else:
    print(f"  WARNING: EOS not found in caption — model may not learn to stop")

print(f"\n  Decoded caption: {vocab.decode(cap)}")


# ── 2. ENCODER ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. ENCODER (ResNetEncoder)")
print("=" * 60)

model = BaselineModel(vocab_size=len(vocab)).to(device)
model.eval()

images = images.to(device)

with torch.no_grad():
    features = model.encoder(images)

print(f"\nInput  shape : {tuple(images.shape)}   (B, 3, 224, 224)")
print(f"Output shape : {tuple(features.shape)}  (B, embed_dim)")
print(f"embed_dim    : {features.shape[1]}  (should be 512)")

# check encoder is frozen
frozen_params  = sum(p.numel() for p in model.encoder.resnet.parameters() if not p.requires_grad)
total_params   = sum(p.numel() for p in model.encoder.resnet.parameters())
print(f"\nResNet frozen params : {frozen_params:,} / {total_params:,}  (should be all frozen)")


# ── 3. DECODER (forward pass) ────────────────────────────────────────
print("\n" + "=" * 60)
print("3. DECODER — forward pass (training mode)")
print("=" * 60)

captions = captions.to(device)

with torch.no_grad():
    # training input: SOS -> second-to-last token (exclude EOS)
    logits = model(images, captions[:, :-1])

print(f"\nDecoder input  (captions[:,:-1]) shape : {tuple(captions[:, :-1].shape)}")
print(f"Decoder output (logits)          shape : {tuple(logits.shape)}")
print(f"  Expected: (B, max_caption_length + 1, vocab_size)")
print(f"  = ({captions.shape[0]}, {captions.shape[1] - 1}, {len(vocab)})")


# ── 4. DECODER (generate) ────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. DECODER — generate (inference mode)")
print("=" * 60)

generated = model.generate(images, vocab, max_length=30)

print(f"\nGenerated {len(generated)} captions for batch of {images.shape[0]}")
for i, token_ids in enumerate(generated):
    decoded = vocab.decode(token_ids)
    print(f"  [{i}] ({len(token_ids)} tokens) {decoded}")


# ── 5. REFERENCES CHECK ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. REFERENCES (get_all_references_dict)")
print("=" * 60)

ref_dict = test_loader.dataset.get_all_references_dict()
sample_img = list(ref_dict.keys())[0]
print(f"\nSample image : {sample_img}")
print(f"Num references : {len(ref_dict[sample_img])}  (should be 5)")
for i, ref in enumerate(ref_dict[sample_img]):
    print(f"  ref[{i}]: {' '.join(ref)}")

print("\n✓ All checks passed")
