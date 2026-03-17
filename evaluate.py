import torch
import argparse
from dataloader import get_flickr8k_loaders
from training_helpers import test_model
from models import BaselineModel

"""
Evaluate a saved model checkpoint on the test set.
Prints BLEU-1 through BLEU-4 scores.

Usage:
    python evaluate.py --checkpoint_path saved_models/models_3_14/model.pt --dataset_dir flickr8k
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(opt):
    # load data
    _, _, test_loader, vocab = get_flickr8k_loaders(root_dir=opt.dataset_dir)

    # build model and load weights from checkpoint
    model = BaselineModel(vocab_size=len(vocab)).to(device)
    checkpoint = torch.load(opt.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {opt.checkpoint_path}")
    print(f"Checkpoint was saved at epoch {checkpoint['epoch']} with val loss {checkpoint['val_loss']:.4f}")

    # run evaluation
    print("\nRunning evaluation on test set...")
    bleu_scores = test_model(model, test_loader, vocab)

    print("\n--- BLEU Scores ---")
    print(f"  BLEU-1: {bleu_scores['bleu1'] * 100:.2f}")
    print(f"  BLEU-2: {bleu_scores['bleu2'] * 100:.2f}")
    print(f"  BLEU-3: {bleu_scores['bleu3'] * 100:.2f}")
    print(f"  BLEU-4: {bleu_scores['bleu4'] * 100:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, 
                        default="C:\\Users\\yashr\\Desktop\\NEU\\Semester 4\\DL\\Project\\CS7150_Image_Captioning_Project\\saved_models\\models_3_14\\model.pt", 
                        required=False, 
                        help="path to saved model checkpoint")
    parser.add_argument("--dataset_dir", type=str, 
                        default="flickr8k", 
                        help="path to flickr8k folder")
    opt = parser.parse_args()

    main(opt)
