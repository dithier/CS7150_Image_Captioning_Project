import torch
import argparse
import math
from collections import defaultdict
from dataloader_v2 import get_flickr8k_loaders

"""
Evaluate a saved model checkpoint using CIDEr-D score.

Implementation note:
    This is a faithful pure-Python port of the CIDEr-D implementation
    in pycocoevalcap/cider/cider_scorer.py (the COCO eval toolkit used
    by the SAT paper). Key details matched exactly:

        - n-gram orders 1 through 4 (n_max=4)
        - IDF formula: log((N - df + 0.5) / (df + 0.5))  [BM25-style]
          This is the exact formula in cider_scorer.py line ~90
        - TF: raw count (NOT normalized by caption length)
          cider_scorer.py uses counts directly, not divided by total
        - Gaussian length penalty: sigma=6.0
        - Clipping: dot product clipped to >= 0 per n-gram
        - Final score scaled by 10.0 (COCO toolkit convention)
        - Scores averaged over all reference captions per image

    This will produce numerically identical results to pycocoevalcap
    for the same tokenized inputs.

No external dependencies beyond PyTorch and your existing codebase.

Usage:
    # for checkpoints without init_c (adam_pass_1, adam_pass_2, adam_pass_4)
    python evaluate_cider.py --checkpoint_path saved_models/adam_pass_4/model.pt --model_version v1

    # for checkpoints with init_c (baseline_v2_adam, sgd_v2 runs)
    python evaluate_cider.py --checkpoint_path saved_models/baseline_v2_adam/model.pt --model_version v2
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# CIDEr-D: exact port of pycocoevalcap
# ──────────────────────────────────────────────

def get_ngrams(tokens, n):
    """
    Count n-grams in a token list.
    Matches cook_refs / cook_test in cider_scorer.py.
    Returns defaultdict(int): ngram_tuple -> count
    """
    counts = defaultdict(int)
    for i in range(len(tokens) - n + 1):
        counts[tuple(tokens[i: i + n])] += 1
    return counts


def compute_doc_freq(crefs, n):
    """
    Document frequency for each n-gram across all reference sets.
    doc_freq[ngram] = number of images where ngram appears in >= 1 reference.

    Matches cider_scorer.py compute_doc_freq().
    """
    doc_freq = defaultdict(int)
    for refs in crefs:
        all_ngrams = set()
        for ref in refs:
            for ngram in get_ngrams(ref, n):
                all_ngrams.add(ngram)
        for ngram in all_ngrams:
            doc_freq[ngram] += 1
    return doc_freq


def compute_cider_n(test_counts, refs_counts, doc_freq, num_images, n, sigma=6.0):
    """
    CIDEr-D score for one image at one n-gram order.

    Matches sim() in cider_scorer.py exactly:
        - IDF = log((N - df + 0.5) / (df + 0.5))   [BM25-style]
        - TF  = raw ngram count (not length-normalised)
        - vec = TF * IDF per ngram
        - score = sum over refs of (dot(hyp, ref) / (norm(hyp)*norm(ref)))
        - clipped to >= 0
        - Gaussian length penalty applied
        - averaged over number of references
    """
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

        dot = sum(
            hyp_vec[ng] * ref_vec[ng]
            for ng in ref_vec
            if ng in hyp_vec
        )
        dot = max(dot, 0.0)

        if hyp_norm * ref_norm > 0:
            score += dot / (hyp_norm * ref_norm)

    penalty = math.exp(
        -((hyp_len - avg_ref_len) ** 2) / (2 * sigma ** 2)
    )

    score = (score / len(refs_counts)) * penalty
    return score


def compute_cider(ref_dict, hypotheses_dict, n_max=4, sigma=6.0):
    """
    Corpus-level CIDEr-D score.

    Matches CiderScorer.compute_score() in pycocoevalcap:
        1. Compute doc frequencies across all reference sets
        2. For each image, score hypothesis vs all references
        3. Average CIDEr-D_{1..4} then average across images
        4. Scale by 10.0

    Parameters
    ----------
    ref_dict : dict { image_name -> list of list of str }
    hypotheses_dict : dict { image_name -> list of str }
    n_max : int  (default 4)
    sigma : float  (default 6.0)

    Returns
    -------
    float — CIDEr-D score on COCO scale (multiply by 100 to compare to papers
            that report as percentage, though most report raw e.g. 85.4)
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

    scores_per_image = []
    for i in range(num_images):
        image_score = 0.0
        for n in range(1, n_max + 1):
            image_score += compute_cider_n(
                test_ngrams[n][i],
                refs_ngrams[n][i],
                doc_freqs[n],
                num_images,
                n,
                sigma,
            )
        image_score /= n_max
        scores_per_image.append(image_score)

    return (sum(scores_per_image) / num_images) * 10.0


def generate_captions(model, data_loader, vocab):
    """
    Run model.generate() over entire dataloader.
    Returns dict: { image_name -> list of str (tokenized words) }
    Each image generated exactly once.

    Note: both baseline_model_v1 and baseline_model_v2 generate() return
    word strings directly (not token indices), so no vocab.decode() needed.
    """
    model.eval()
    all_generated = {}

    with torch.no_grad():
        for images, captions, image_names in data_loader:
            images = images.to(device)
            batch_generated = model.generate(images, vocab)

            for img_name, words in zip(image_names, batch_generated):
                if img_name not in all_generated:
                    all_generated[img_name] = words if words else ["<unk>"]

    return all_generated


def main(opt):
    # Load data
    _, _, test_loader, vocab = get_flickr8k_loaders(root_dir=opt.dataset_dir)

    # Load correct model class
    if opt.model_version == "v1":
        from baseline_model_v1 import BaselineModel
        print("Using baseline_model_v1 (no init_c, c0 = zeros)")
    else:
        from baseline_model_v2 import BaselineModel
        print("Using baseline_model_v2 (with learned init_c)")

    model = BaselineModel(vocab_size=len(vocab)).to(device)
    checkpoint = torch.load(opt.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {opt.checkpoint_path}")
    print(f"Saved at epoch {checkpoint['epoch']} | val loss {checkpoint['val_loss']:.4f}")

    # Generate
    print("\nGenerating captions over test set...")
    hypotheses_dict = generate_captions(model, test_loader, vocab)

    # References
    ref_dict = test_loader.dataset.get_all_references_dict()

    # Sanity checks
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

    # Score
    print("Computing CIDEr-D (may take ~30s)...")
    cider = compute_cider(ref_dict, hypotheses_dict)

    print("\n--- CIDEr-D Score ---")
    print(f"  CIDEr: {cider:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="path to saved model checkpoint")
    parser.add_argument("--model_version", type=str,
                        choices=["v1", "v2"], required=True,
                        help="v1 = no init_c (adam_pass_1/2/4), v2 = with init_c (baseline_v2_adam, sgd_v2)")
    parser.add_argument("--dataset_dir", type=str, default="flickr8k",
                        help="path to flickr8k folder")
    opt = parser.parse_args()
    main(opt)
