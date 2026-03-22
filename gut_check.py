import torch
import argparse
from dataloader import get_flickr8k_loaders
from models import BaselineModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_results(model, vocab, image, caption, file):
    # image is C X W X H
    image = image.unsqueeze(0) # 1 X C X W X H

    english_caption = [vocab.index_to_word[x] for x in caption.tolist()]

    print(f"Image: {file}\nOne caption: {english_caption}")

    output = model.generate(image, vocab)
    print(f"model output: {output}")
    print("\n\n")

def main(opt):
    # load data
    train_loader, val_loader, test_loader, vocab = get_flickr8k_loaders(root_dir=opt.dataset_dir)

    train_image, train_caption, train_file = train_loader.dataset[0]
    test_image, test_caption, test_file = test_loader.dataset[0]

    # build model and load weights from checkpoint
    model = BaselineModel(vocab_size=len(vocab)).to(device)
    checkpoint = torch.load(opt.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {opt.checkpoint_path}")
    print(f"Checkpoint was saved at epoch {checkpoint['epoch']} with val loss {checkpoint['val_loss']:.4f}")

    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(name)

    print("Train example")
    get_results(model, vocab, train_image, train_caption, train_file)

    print("Test example")
    get_results(model, vocab, test_image, test_caption, test_file)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, 
                        default="model.pt", 
                        required=False, 
                        help="path to saved model checkpoint")
    parser.add_argument("--dataset_dir", type=str, 
                        default="flickr8k", 
                        help="path to flickr8k folder")
    opt = parser.parse_args()

    main(opt)