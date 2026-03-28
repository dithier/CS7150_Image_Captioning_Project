import torch
import argparse
from dataloader_v2 import get_flickr8k_loaders
from eval_metrics import evaluation_metric
from training_helpers_1 import test_model_v1, test_model_v2, test_model_from_models

"""
Evaluate a saved model checkpoint on the test set.
Prints BLEU-1 through BLEU-4 scores.

Usage:
    # for checkpoints without init_c (adam_pass_1, adam_pass_2, adam_pass_4)
    python evaluate.py --checkpoint_path saved_models/adam_pass_4/model.pt --model_version v1

    # for checkpoints with init_c (baseline_v2_adam, sgd_v2 runs)
    python evaluate.py --checkpoint_path saved_models/baseline_v2_adam/model.pt --model_version v2
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(opt):
    batch_size = opt.ref * 32

    # load data
    _, _, test_loader, vocab = get_flickr8k_loaders(root_dir=opt.dataset_dir, batch_size=batch_size)

    # load correct model class based on version
    if opt.model_version == "v1":
        from baseline_model_v1 import BaselineModel
        test_fn = test_model_v1
        print("Using baseline_model_v1 (no init_c, c0 = zeros)")
    elif opt.model_version == "v2":
        from baseline_model_v2 import BaselineModel
        test_fn = test_model_v2
        print("Using baseline_model_v2 (with learned init_c)")
    else:
        from models import BaselineModel
        test_fn = test_model_from_models
        print("Using models.py BaselineModel (returns token indices)")

    # build model and load weights from checkpoint
    model = BaselineModel(vocab_size=len(vocab)).to(device)
    checkpoint = torch.load(opt.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {opt.checkpoint_path}")
    print(f"Checkpoint was saved at epoch {checkpoint['epoch']} with val loss {checkpoint['val_loss']:.4f}")

    # run evaluation
    print("\nRunning evaluation on test set...")
    bleu_scores = test_fn(model, test_loader, vocab)

    print("\n--- BLEU Scores ---")
    print(f"  BLEU-1: {bleu_scores['bleu1'] * 100:.2f}")
    print(f"  BLEU-2: {bleu_scores['bleu2'] * 100:.2f}")
    print(f"  BLEU-3: {bleu_scores['bleu3'] * 100:.2f}")
    print(f"  BLEU-4: {bleu_scores['bleu4'] * 100:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str,
                        required=True,
                        help="path to saved model checkpoint")
    parser.add_argument("--model_version", type=str,
                        choices=["v1", "v2", "models"],
                        required=True,
                        help="v1 = baseline_model_v1 (no init_c), v2 = baseline_model_v2 (with init_c), models = models.py")
    parser.add_argument("--dataset_dir", type=str,
                        default="flickr8k",
                        help="path to flickr8k folder")
    parser.add_argument("--ref", type=int, default=5,
                        help="number of references per image")
    opt = parser.parse_args()

    main(opt)
