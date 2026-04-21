"""
save_vocab.py

Run this ONCE on Discovery to generate vocab.pkl.
vocab.pkl is then used by the attention visualization scripts.

Usage:
    python save_vocab.py --root ./flickr8k
"""

import argparse
import pickle
from dataloader_v2 import get_flickr8k_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, required=True, help="Path to flickr8k/ folder")
parser.add_argument("--output", type=str, default="vocab.pkl", help="Where to save vocab")
args = parser.parse_args()

_, _, _, vocab = get_flickr8k_loaders(root_dir=args.root)

with open(args.output, "wb") as f:
    pickle.dump(vocab, f)

print(f"Saved {args.output} — {len(vocab)} tokens")
