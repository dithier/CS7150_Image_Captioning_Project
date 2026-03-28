import torch
import argparse
from dataloader_v2 import get_flickr8k_loaders

"""
Evaluate a saved model checkpoint using METEOR score.

Implementation note:
    The SAT paper (Xu et al. 2015) used the official METEOR 1.5 Java tool.
    This script uses nltk.translate.meteor_score, which implements the same
    algorithm (unigram F-score with stemming + WordNet synonym matching),
    and matches the pycocoevalcap METEOR wrapper as closely as possible in
    pure Python:

        - alpha = 0.9   (harmonic mean weight, same as METEOR 1.5 default)
        - beta  = 3.0   (penalty weight)
        - gamma = 0.5   (fragmentation penalty)

    These are the exact defaults used by the COCO eval toolkit's METEOR
    wrapper (pycocoevalcap/meteor/meteor.py).

    Scores will be within ~1-2 points of the official Java tool due to
    minor tokenization differences, but this is the closest pure-Python
    equivalent.

Requires:
    pip install nltk
    python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

Usage:
    # for checkpoints without init_c (adam_pass_1, adam_pass_2, adam_pass_4)
    python evaluate_meteor.py --checkpoint_path saved_models/adam_pass_4/model.pt --model_version v1

    # for checkpoints with init_c (baseline_v2_adam, sgd_v2 runs)
    python evaluate_meteor.py --checkpoint_path saved_models/baseline_v2_adam/model.pt --model_version v2
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_meteor(ref_dict, hypotheses_dict):
    """
    Corpus-level METEOR following pycocoevalcap convention:
        - Score each image against all 5 references
        - Average across the corpus

    Parameters
    ----------
    ref_dict : dict { image_name -> list of list of str }
        5 tokenized reference captions per image.
        From dataset.get_all_references_dict()

    hypotheses_dict : dict { image_name -> list of str }
        One tokenized generated caption per image.

    Returns
    -------
    float — mean METEOR (0 to 1). Multiply by 100 for percentage.
    """
    from nltk.translate.meteor_score import meteor_score

    # METEOR 1.5 defaults — same as pycocoevalcap
    scores = []
    for img_name, references in ref_dict.items():
        hypothesis = hypotheses_dict.get(img_name, ["<unk>"])

        # nltk meteor_score matches METEOR 1.5:
        #   alpha=0.9, beta=3, gamma=0.5
        score = meteor_score(
            references,
            hypothesis,
            alpha=0.9,
            beta=3.0,
            gamma=0.5
        )
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


def generate_captions(model, data_loader, vocab):
    """
    Run model.generate() over entire dataloader.
    Returns dict: { image_name -> list of str (tokenized words) }
    Each image generated exactly once (deduplication).

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
                    # words is already a list of strings
                    all_generated[img_name] = words if words else ["<unk>"]

    return all_generated


def main(opt):
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    # Load data
    _, _, test_loader, vocab = get_flickr8k_loaders(root_dir=opt.dataset_dir)

    # Load correct model class based on version
    if opt.model_version == "v1":
        from baseline_model_v1 import BaselineModel
        print("Using baseline_model_v1 (no init_c, c0 = zeros)")
    else:
        from baseline_model_v2 import BaselineModel
        print("Using baseline_model_v2 (with learned init_c)")

    # Load model
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

    # Score
    print("Computing METEOR...")
    meteor = compute_meteor(ref_dict, hypotheses_dict)

    print("\n--- METEOR Score ---")
    print(f"  METEOR: {meteor * 100:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="path to saved model checkpoint")
    parser.add_argument("--model_version", type=str,
                        choices=["v1", "v2"],
                        required=True,
                        help="v1 = no init_c (adam_pass_1/2/4), v2 = with init_c (baseline_v2_adam, sgd_v2)")
    parser.add_argument("--dataset_dir", type=str, default="flickr8k",
                        help="path to flickr8k folder")
    opt = parser.parse_args()
    main(opt)
