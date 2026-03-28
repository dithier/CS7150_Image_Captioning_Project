import torch
import argparse
from dataloader_v2 import get_flickr8k_loaders
from eval_metrics import evaluation_metric

"""
Evaluate a saved model checkpoint on the test set.
Supports BLEU-1 through BLEU-4 and METEOR scores.

Usage:
    # all metrics (default)
    python evaluate_all.py --checkpoint_path saved_models/adam_pass_4/model.pt --model_version v1

    # bleu only
    python evaluate_all.py --checkpoint_path saved_models/adam_pass_4/model.pt --model_version v1 --metric bleu

    # meteor only
    python evaluate_all.py --checkpoint_path saved_models/baseline_v2_adam/model.pt --model_version v2 --metric meteor
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_captions(model, data_loader, vocab):
    """
    Run model.generate() over entire dataloader.
    Returns dict: { image_name -> list of str }
    Both v1 and v2 generate() return word strings directly.
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


def compute_bleu(data_loader, hypotheses_dict, ref_dict):
    """
    Compute BLEU-1 through BLEU-4 using corpus_bleu.
    hypotheses must be in same order as ref_dict keys.
    """
    hypotheses = [hypotheses_dict[img] for img in ref_dict.keys()]
    bleu_scores = evaluation_metric(data_loader, hypotheses)
    return bleu_scores


def compute_meteor(ref_dict, hypotheses_dict):
    """
    Corpus-level METEOR following pycocoevalcap convention.
    Uses METEOR 1.5 defaults: alpha=0.9, beta=3.0, gamma=0.5
    """
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


def main(opt):
    # download wordnet if meteor is needed
    if opt.metric in ("meteor", "all"):
        import nltk
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    # load data
    _, _, test_loader, vocab = get_flickr8k_loaders(
        root_dir=opt.dataset_dir,
        batch_size=opt.ref * 32  # keeps all refs for an image in same batch
    )

    # load correct model class
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

    # generate captions once — reuse for both metrics
    print("\nGenerating captions over test set...")
    hypotheses_dict = generate_captions(model, test_loader, vocab)
    ref_dict = test_loader.dataset.get_all_references_dict()

    # sanity check
    assert len(hypotheses_dict) == len(ref_dict), (
        f"Mismatch: {len(hypotheses_dict)} hypotheses vs {len(ref_dict)} reference images"
    )
    empty = sum(1 for h in hypotheses_dict.values() if h == ["<unk>"])
    if empty:
        print(f"Warning: {empty} empty hypotheses replaced with <unk>")

    # compute requested metrics
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="path to saved model checkpoint")
    parser.add_argument("--model_version", type=str,
                        choices=["v1", "v2"], required=True,
                        help="v1 = no init_c (adam_pass_1/2/4), v2 = with init_c (baseline_v2_adam, sgd_v2)")
    parser.add_argument("--metric", type=str,
                        choices=["bleu", "meteor", "all"], default="all",
                        help="which metric to compute (default: all)")
    parser.add_argument("--dataset_dir", type=str, default="flickr8k",
                        help="path to flickr8k folder")
    parser.add_argument("--ref", type=int, default=5,
                        help="number of references per image")
    opt = parser.parse_args()

    main(opt)
