import torch
import argparse
import math
import nltk
from collections import defaultdict
from dataloader_v2 import get_flickr8k_loaders
from nltk.translate.bleu_score import corpus_bleu
from training_helpers import evaluate, evaluateRandomly

"""
Unified evaluation script for all model types.
Supports BLEU-1 through BLEU-4, METEOR, and CIDEr-D.

Model versions:
  baseline_v2         - BaselineModel with learned c0 (LSTM baseline)
  resnet_transformer  - ResnetTransformerModel (pretrained ResNet50 + transformer decoder)
  vit_transformer     - VisionTransformerModel (pretrained ViT-B/16 + transformer decoder)
    todo: fix model paths
Usage:
  python evaluate_all.py --model_version baseline_v2 --checkpoint_path saved_models/sgd_v2/model.pt
  python evaluate_all.py --model_version baseline_v2 --checkpoint_path ... --metric bleu

  python evaluate_all.py --model_version resnet_transformer --checkpoint_path saved_models/resnet_transformer_pass_1/model.pt

  python evaluate_all.py --model_version vit_transformer --checkpoint_path saved_models/vit_transformer_pass_1/model.pt
  python evaluate_all.py --model_version vit_transformer --checkpoint_path ... --metric cider
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────

def load_model(opt, vocab):
    """
    Load and return the correct model class based on --model_version.
    """
    if opt.model_version == "baseline_v2":
        from baseline.baseline_model_v2 import BaselineModel
        print("Using baseline_model_v2 (LSTM with learned c0)")
        # todo: are default params def what we used?
        model = BaselineModel(vocab_size=len(vocab)).to(device)

    elif opt.model_version == "decoder_only":
        from resnet_transformer_decoder.resnet_transformer import ResnetTransformerModel
        print("Using Resnet Transformer Decoder")
        # todo: are these the params we used?
        embed_dim = 256
        num_heads = 8
        trx_ff_dim = 1024      # 4 * embed_dim
        num_decoder_cells = 3
        dropout = 0.3
        freeze = True

        model = ResnetTransformerModel(vocab, num_heads, trx_ff_dim, num_decoder_cells,
                                        embed_dim, dropout, freeze).to(device)
    


    elif opt.model_version == "vit_transformer":
        from ViT_decoder.pytorch_pretrainined_enc_dec_model import VisionTransformerModel
        print("Using VisionTransformerModel (pretrained ViT-B/16 + transformer decoder)")
        # todo: are these the params we used?
        model = VisionTransformerModel(vocab).to(device)

    else:
        raise ValueError(f"Unknown model_version: {opt.model_version}")

    checkpoint = torch.load(opt.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {opt.checkpoint_path}")
    print(f"Saved at epoch {checkpoint['epoch']} | val loss {checkpoint['val_loss']:.4f}")

    return model


# ──────────────────────────────────────────────
# Caption Generation
# ──────────────────────────────────────────────

def generate_captions_word_strings(model, data_loader, vocab):
    """
    For models whose generate() returns word strings directly.
    - baseline_v2: model.generate(images, vocab)
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


def generate_captions_from_logits(model, data_loader, vocab):
    model.eval()
    all_generated = {}
    eos_idx = vocab.word_to_index[vocab.EOS_TOKEN]
    pad_idx = vocab.word_to_index[vocab.PAD_TOKEN]
    sos_idx = vocab.word_to_index[vocab.SOS_TOKEN]

    with torch.no_grad():
        for batch_idx, (images, captions, image_names) in enumerate(data_loader):
            images = images.to(device)
            print(f"  Generating batch {batch_idx + 1}/{len(data_loader)}...", flush=True)
            logits = model.forward(images)
            predicted = logits.argmax(dim=-1)

            for img_name, token_ids in zip(image_names, predicted):
                words = []
                for tok in token_ids.tolist():
                    if tok in (eos_idx, pad_idx, sos_idx):
                        break
                    words.append(vocab.index_to_word.get(tok, "<unk>"))
                if img_name not in all_generated:
                    all_generated[img_name] = words if words else ["<unk>"]

    return all_generated


def generate_captions(model_version, model, data_loader, vocab):
    """
    Route to the correct generation function based on model type.
    """
    if model_version == "baseline_v2":
        return generate_captions_word_strings(model, data_loader, vocab=vocab)
    elif model_version in ["vit_transformer, resnet_transformer"]:
        # resnet transformer or vit transformer
        return generate_captions_from_logits(model, data_loader, vocab=vocab)
    else:
        raise ValueError(f"Unknown model_version: {model_version}")


# ──────────────────────────────────────────────
# BLEU
# ──────────────────────────────────────────────

def compute_bleu(ref_dict, hypotheses_dict):
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
# CIDEr-D
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

    scores_per_order = {n: [] for n in range(1, n_max + 1)}
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
            n_score = (score / len(refs_ngrams[n][i])) * penalty
            scores_per_order[n].append(n_score)
            image_score += n_score

        scores_per_image.append(image_score / n_max)

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
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    _, _, test_loader, vocab = get_flickr8k_loaders(
        root_dir=opt.dataset_dir,
        batch_size=opt.ref * 32
    )

    model = load_model(opt, vocab)

    print("\nGenerating captions over test set...")
    hypotheses_dict = generate_captions(opt.model_version, model, test_loader, vocab)
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

    print(f"\n========== Evaluation Results [{opt.model_version}] ==========")

    if opt.metric in ("bleu", "all"):
        print("\nComputing BLEU scores...")
        bleu = compute_bleu(ref_dict, hypotheses_dict)
        print(f"  BLEU-1: {bleu['bleu1'] * 100:.2f}")
        print(f"  BLEU-2: {bleu['bleu2'] * 100:.2f}")
        print(f"  BLEU-3: {bleu['bleu3'] * 100:.2f}")
        print(f"  BLEU-4: {bleu['bleu4'] * 100:.2f}")

    if opt.metric in ("meteor", "all"):
        print("\nComputing METEOR score...")
        meteor = compute_meteor(ref_dict, hypotheses_dict)
        print(f"  METEOR: {meteor * 100:.2f}")

    if opt.metric in ("cider", "all"):
        print("\nComputing CIDEr-D (may take ~30s)...")
        overall, per_order = compute_cider(ref_dict, hypotheses_dict)
        print(f"  CIDEr-D (overall): {overall:.2f}")
        for n in range(1, 5):
            print(f"  CIDEr-{n}          : {per_order[n]:.2f}")

    print("=====================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="path to saved model checkpoint")
    parser.add_argument("--model_version", type=str, required=True,
                        choices=["baseline_v2", "resnet_transformer",  "vit_transformer"],
                        help="which model to evaluate")
    parser.add_argument("--metric", type=str, default="all",
                        choices=["bleu", "meteor", "cider", "all"],
                        help="which metric(s) to compute (default: all)")
    parser.add_argument("--dataset_dir", type=str, default="flickr8k",
                        help="path to flickr8k folder")
    parser.add_argument("--ref", type=int, default=5,
                        help="number of references per image")
    opt = parser.parse_args()

    main(opt)