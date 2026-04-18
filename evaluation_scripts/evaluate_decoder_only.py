import torch
import argparse
import math
from collections import defaultdict
from dataloader_v2 import get_flickr8k_loaders
from evaluation_scripts.get_bleu_score import evaluation_metric
from decoder_transformer_only_model import VisionTransformerDecoderModel  # CHANGED

"""
Evaluate a saved decoder-only transformer checkpoint on the test set.
Supports BLEU-1 through BLEU-4, METEOR, CIDEr-D, and CIDEr per n-gram order.

Usage:
    # all metrics (default)
    python evaluate_decoder_transformer.py --checkpoint_path saved_models/decoder_transformer_pass_1/model.pt

    # bleu only
    python evaluate_decoder_transformer.py --checkpoint_path saved_models/decoder_transformer_pass_1/model.pt --metric bleu

    # meteor only
    python evaluate_decoder_transformer.py --checkpoint_path saved_models/decoder_transformer_pass_1/model.pt --metric meteor

    # cider only
    python evaluate_decoder_transformer.py --checkpoint_path saved_models/decoder_transformer_pass_1/model.pt --metric cider
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# Caption Generation
# ──────────────────────────────────────────────

def generate_captions(model, data_loader):  # CHANGED: no vocab arg — decoder model doesn't need it
    """
    Run model.generate() over entire dataloader.
    Returns dict: { image_name -> list of str }
    Decoder-only generate() returns word strings directly (no vocab needed).
    """
    model.eval()
    all_generated = {}

    with torch.no_grad():
        for images, captions, image_names in data_loader:
            images = images.to(device)
            batch_generated = model.generate(images)  # CHANGED: no vocab arg

            for img_name, words in zip(image_names, batch_generated):
                if img_name not in all_generated:
                    all_generated[img_name] = words if words else ["<unk>"]

    return all_generated


# ──────────────────────────────────────────────
# BLEU
# ──────────────────────────────────────────────

def compute_bleu(data_loader, hypotheses_dict, ref_dict):
    hypotheses = [hypotheses_dict[img] for img in ref_dict.keys()]
    return evaluation_metric(data_loader, hypotheses)


# ──────────────────────────────────────────────
# METEOR
# ──────────────────────────────────────────────

def compute_meteor(ref_dict, hypotheses_dict):
    from nltk.translate.meteor_score import meteor_score

    scores = []
    for img_name, references in ref_dict.items():
        hypothesis = hypotheses_dict.get(img_name, ["<unk>"])
        score = meteor_score(
            references,
            hypothesis,
            alpha=0.9,
            beta=3.0,
            gamma=0.5
        )
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


# ──────────────────────────────────────────────
# CIDEr-D
# ──────────────────────────────────────────────

def get_ngrams(tokens, n):
    counts = defaultdict(int)
    for i in range(len(tokens) - n + 1):
        counts[tuple(tokens[i: i + n])] += 1
    return counts


def compute_cider_n(test_counts, refs_counts, doc_freq, num_images, n, sigma=6.0):
    def tfidf_vec(counts):
        vec = defaultdict(float)
        norm = 0.0
        for ngram, count in counts.items():
            df = doc_freq.get(ngram, 0)
            idf = math.log((num_images - df + 0.5) / (df + 0.5))
            vec[ngram] = count * idf
            norm += (count * idf) ** 2
        norm = math.sqrt(norm)
        return vec, norm

    hyp_vec, hyp_norm = tfidf_vec(test_counts)

    ref_lens = [sum(v for v in rc.values()) for rc in refs_counts]
    avg_ref_len = sum(ref_lens) / len(ref_lens) if ref_lens else 1
    hyp_len = sum(v for v in test_counts.values())

    score = 0.0
    for ref_counts in refs_counts:
        ref_vec, ref_norm = tfidf_vec(ref_counts)
        dot = sum(hyp_vec[ng] * ref_vec[ng] for ng in ref_vec if ng in hyp_vec)
        dot = max(dot, 0.0)
        if hyp_norm * ref_norm > 0:
            score += dot / (hyp_norm * ref_norm)

    penalty = math.exp(-((hyp_len - avg_ref_len) ** 2) / (2 * sigma ** 2))
    return (score / len(refs_counts)) * penalty


def compute_cider(ref_dict, hypotheses_dict, n_max=4, sigma=6.0):
    """
    Returns overall CIDEr-D score and per n-gram order breakdown.
    """
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

    scores_per_order = {n: [] for n in range(1, n_max + 1)}
    scores_per_image = []

    for i in range(num_images):
        image_score = 0.0
        for n in range(1, n_max + 1):
            n_score = compute_cider_n(
                test_ngrams[n][i],
                refs_ngrams[n][i],
                doc_freqs[n],
                num_images,
                n,
                sigma,
            )
            scores_per_order[n].append(n_score)
            image_score += n_score
        image_score /= n_max
        scores_per_image.append(image_score)

    overall = (sum(scores_per_image) / num_images) * 10.0
    per_order = {
        n: (sum(scores_per_order[n]) / num_images) * 10.0
        for n in range(1, n_max + 1)
    }

    return overall, per_order


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main(opt):
    if opt.metric in ("meteor", "all"):
        import nltk
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    _, _, test_loader, vocab = get_flickr8k_loaders(
        root_dir=opt.dataset_dir,
        batch_size=opt.ref * 32
    )

    # CHANGED: decoder-only model instantiation, no num_encoder_cells
    P = 16
    embed_dim = 256
    num_heads = 8
    trx_ff_dim = embed_dim * 4
    num_decoder_cells = 4#6
    dropout = 0.3 #0.1

    model = VisionTransformerDecoderModel(
        vocab, P, embed_dim, num_heads, trx_ff_dim, num_decoder_cells, dropout
    ).to(device)

    checkpoint = torch.load(opt.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {opt.checkpoint_path}")
    print(f"Saved at epoch {checkpoint['epoch']} | val loss {checkpoint['val_loss']:.4f}")

    # generate once — reuse for all metrics
    print("\nGenerating captions over test set...")
    hypotheses_dict = generate_captions(model, test_loader)  # CHANGED: no vocab arg
    ref_dict = test_loader.dataset.get_all_references_dict()

    assert len(hypotheses_dict) == len(ref_dict), (
        f"Mismatch: {len(hypotheses_dict)} hypotheses vs {len(ref_dict)} reference images"
    )
    empty = sum(1 for h in hypotheses_dict.values() if h == ["<unk>"])
    if empty:
        print(f"Warning: {empty} empty hypotheses replaced with <unk>")

    if opt.metric in ("bleu", "all"):
        print("\nComputing BLEU scores...")
        bleu = compute_bleu(test_loader, hypotheses_dict, ref_dict)
        print("\n--- BLEU Scores ---")
        print(f"  BLEU-1: {bleu['bleu1'] * 100:.2f}")
        print(f"  BLEU-2: {bleu['bleu2'] * 100:.2f}")
        print(f"  BLEU-3: {bleu['bleu3'] * 100:.2f}")
        print(f"  BLEU-4: {bleu['bleu4'] * 100:.2f}")

    if opt.metric in ("meteor", "all"):
        print("\nComputing METEOR score...")
        meteor = compute_meteor(ref_dict, hypotheses_dict)
        print("\n--- METEOR Score ---")
        print(f"  METEOR: {meteor * 100:.2f}")

    if opt.metric in ("cider", "all"):
        print("\nComputing CIDEr-D (may take ~30s)...")
        overall, per_order = compute_cider(ref_dict, hypotheses_dict)
        print("\n--- CIDEr-D Scores ---")
        print(f"  CIDEr-D (overall): {overall:.2f}")
        for n in range(1, 5):
            print(f"  CIDEr-{n}          : {per_order[n]:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="path to saved decoder transformer checkpoint")
    parser.add_argument("--metric", type=str,
                        choices=["bleu", "meteor", "cider", "all"], default="all",
                        help="which metric(s) to compute (default: all)")
    parser.add_argument("--dataset_dir", type=str, default="flickr8k",
                        help="path to flickr8k folder")
    parser.add_argument("--ref", type=int, default=5,
                        help="number of references per image")
    opt = parser.parse_args()
    main(opt)