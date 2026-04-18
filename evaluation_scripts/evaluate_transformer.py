import torch
import argparse
import math
import nltk
from collections import defaultdict
from dataloader_v2 import get_flickr8k_loaders
from transformer_model import VisionTransformerModel
from nltk.translate.bleu_score import corpus_bleu

"""
Evaluate a saved transformer model checkpoint on the test set.
Reports BLEU-1 through BLEU-4, METEOR, and CIDEr-D.

Usage:
    python evaluate_transformer.py \
        --checkpoint_path saved_models/diy_transformer_pass_1/model.pt \
        --dataset_dir flickr8k
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# Caption generation
# ──────────────────────────────────────────────

def generate_captions(model, data_loader):
    """
    Run model.generate() over the entire dataloader.
    Returns dict: { image_name -> list of str (tokenized words) }
    Each image generated exactly once.

    NOTE: transformer generate() returns word strings directly (not token ids),
    so we just use the list as-is.
    """
    model.eval()
    all_generated = {}

    with torch.no_grad():
        for images, captions, image_names in data_loader:
            images = images.to(device)
            batch_generated = model.generate(images)  # list of list of str

            for img_name, words in zip(image_names, batch_generated):
                if img_name not in all_generated:
                    all_generated[img_name] = words if words else ["<unk>"]

    return all_generated


# ──────────────────────────────────────────────
# BLEU
# ──────────────────────────────────────────────

def compute_bleu(ref_dict, hypotheses_dict):
    """
    Corpus BLEU-1 through BLEU-4.
    ref_dict: { image_name -> list of list of str }
    hypotheses_dict: { image_name -> list of str }
    """
    references = []
    hypotheses = []

    for img_name, refs in ref_dict.items():
        references.append(refs)
        hypotheses.append(hypotheses_dict.get(img_name, ["<unk>"]))

    bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    return {"bleu1": bleu1, "bleu2": bleu2, "bleu3": bleu3, "bleu4": bleu4}


# ──────────────────────────────────────────────
# METEOR
# ──────────────────────────────────────────────

def compute_meteor(ref_dict, hypotheses_dict):
    from nltk.translate.meteor_score import meteor_score

    scores = []
    for img_name, references in ref_dict.items():
        hypothesis = hypotheses_dict.get(img_name, ["<unk>"])
        score = meteor_score(references, hypothesis, alpha=0.9, beta=3.0, gamma=0.5)
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


# ──────────────────────────────────────────────
# CIDEr-D (exact port of pycocoevalcap)
# ──────────────────────────────────────────────

def get_ngrams(tokens, n):
    counts = defaultdict(int)
    for i in range(len(tokens) - n + 1):
        counts[tuple(tokens[i: i + n])] += 1
    return counts


def compute_cider(ref_dict, hypotheses_dict, n_max=4, sigma=6.0):
    image_names = list(ref_dict.keys())
    num_images = len(image_names)

    refs_ngrams = {n: [] for n in range(1, n_max + 1)}
    test_ngrams = {n: [] for n in range(1, n_max + 1)}

    for img_name in image_names:
        refs = ref_dict[img_name]
        hyp = hypotheses_dict.get(img_name, ["<unk>"])
        for n in range(1, n_max + 1):
            refs_ngrams[n].append([get_ngrams(ref, n) for ref in refs])
            test_ngrams[n].append(get_ngrams(hyp, n))

    doc_freqs = {}
    for n in range(1, n_max + 1):
        df = defaultdict(int)
        for i in range(num_images):
            seen = set()
            for ref_counts in refs_ngrams[n][i]:
                for ngram in ref_counts:
                    if ngram not in seen:
                        df[ngram] += 1
                        seen.add(ngram)
        doc_freqs[n] = df

    def tfidf_vec(counts, doc_freq):
        vec = defaultdict(float)
        norm = 0.0
        for ngram, count in counts.items():
            df = doc_freq.get(ngram, 0)
            idf = math.log((num_images - df + 0.5) / (df + 0.5))
            vec[ngram] = count * idf
            norm += (count * idf) ** 2
        return vec, math.sqrt(norm)

    scores_per_image = []
    for i in range(num_images):
        image_score = 0.0
        for n in range(1, n_max + 1):
            hyp_vec, hyp_norm = tfidf_vec(test_ngrams[n][i], doc_freqs[n])
            ref_lens = [sum(v for v in rc.values()) for rc in refs_ngrams[n][i]]
            avg_ref_len = sum(ref_lens) / len(ref_lens) if ref_lens else 1
            hyp_len = sum(v for v in test_ngrams[n][i].values())

            score = 0.0
            for ref_counts in refs_ngrams[n][i]:
                ref_vec, ref_norm = tfidf_vec(ref_counts, doc_freqs[n])
                dot = sum(hyp_vec[ng] * ref_vec[ng] for ng in ref_vec if ng in hyp_vec)
                dot = max(dot, 0.0)
                if hyp_norm * ref_norm > 0:
                    score += dot / (hyp_norm * ref_norm)

            penalty = math.exp(-((hyp_len - avg_ref_len) ** 2) / (2 * sigma ** 2))
            image_score += (score / len(refs_ngrams[n][i])) * penalty

        scores_per_image.append(image_score / n_max)

    return (sum(scores_per_image) / num_images) * 10.0


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main(opt):
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    # Load data
    _, _, test_loader, vocab = get_flickr8k_loaders(root_dir=opt.dataset_dir)

    # Build model with same hyperparams used during training
    P = 16
    embed_dim = 256
    num_heads = 8
    trx_ff_dim = embed_dim * 4
    num_encoder_cells = 6
    num_decoder_cells = 6
    dropout = 0.1

    model = VisionTransformerModel(
        vocab, P, embed_dim, num_heads, trx_ff_dim,
        num_encoder_cells, num_decoder_cells, dropout
    ).to(device)

    checkpoint = torch.load(opt.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {opt.checkpoint_path}")
    print(f"Saved at epoch {checkpoint['epoch']} | val loss {checkpoint['val_loss']:.4f}")

    # Generate
    print("\nGenerating captions over test set...")
    hypotheses_dict = generate_captions(model, test_loader)

    # References
    ref_dict = test_loader.dataset.get_all_references_dict()

    assert len(hypotheses_dict) == len(ref_dict), (
        f"Mismatch: {len(hypotheses_dict)} hypotheses vs {len(ref_dict)} reference images"
    )
    empty = sum(1 for h in hypotheses_dict.values() if h == ["<unk>"])
    if empty:
        print(f"Warning: {empty} empty hypotheses replaced with <unk>")

    # Debug sample
    sample_imgs = list(ref_dict.keys())[:3]
    print("\n--- Debug Sample ---")
    for img in sample_imgs:
        print(f"  image : {img}")
        print(f"  hyp   : {hypotheses_dict[img]}")
        print(f"  ref[0]: {ref_dict[img][0]}")
    print("--------------------\n")

    # BLEU
    print("Computing BLEU...")
    bleu = compute_bleu(ref_dict, hypotheses_dict)

    # METEOR
    print("Computing METEOR...")
    meteor = compute_meteor(ref_dict, hypotheses_dict)

    # CIDEr
    print("Computing CIDEr-D (may take ~30s)...")
    cider = compute_cider(ref_dict, hypotheses_dict)

    print("\n========== Transformer Evaluation Results ==========")
    print(f"  BLEU-1 : {bleu['bleu1'] * 100:.2f}")
    print(f"  BLEU-2 : {bleu['bleu2'] * 100:.2f}")
    print(f"  BLEU-3 : {bleu['bleu3'] * 100:.2f}")
    print(f"  BLEU-4 : {bleu['bleu4'] * 100:.2f}")
    print(f"  METEOR : {meteor * 100:.2f}")
    print(f"  CIDEr  : {cider:.2f}")
    print("=====================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str,
                        default="saved_models/diy_transformer_pass_1/model.pt",
                        help="path to saved transformer checkpoint")
    parser.add_argument("--dataset_dir", type=str,
                        default="flickr8k",
                        help="path to flickr8k folder")
    opt = parser.parse_args()
    main(opt)