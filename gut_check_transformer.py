import torch
import argparse
from dataloader_v2 import get_flickr8k_loaders
from pytorch_transformer_enc_dec_model import VisionTransformerModel
from training_helpers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_results(model, vocab, image, caption, file):
    # image is C X W X H
    image = image.unsqueeze(0) # 1 X C X W X H

    english_caption = [vocab.index_to_word[x] for x in caption.tolist() if x is not vocab.word_to_index[vocab.PAD_TOKEN]]
    english_caption = " ".join(english_caption)

    print(f"Image: {file}\nOne caption: {english_caption}")

    output_words = evaluate(model, image, vocab)
    output_sentence = ' '.join(output_words)

    print(f"model output: {output_sentence}")
    print("\n\n")

def main(opt):
    # load data
    train_loader, val_loader, test_loader, vocab = get_flickr8k_loaders(root_dir=opt.dataset_dir)

    train_image, train_caption, train_file = train_loader.dataset[0]
    test_image, test_caption, test_file = test_loader.dataset[0]
    val_image, val_caption, val_file = val_loader.dataset[0]

    # build model and load weights from checkpoint
    # model = BaselineModel(vocab_size=len(vocab)).to(device)

     # Note these model params may be too larger for such a small dataset, Might overfit
    P = 16
    embed_dim = 256 # hidden size D in vision transformer?
    num_heads = 8 
    trx_ff_dim = embed_dim * 4  # couldn't find in paper
    num_encoder_cells = 6 # ViT Base
    num_decoder_cells = 6 # ViT Base
    dropout = 0.1

    model = VisionTransformerModel(vocab, P, embed_dim, num_heads, trx_ff_dim,
                                   num_encoder_cells, num_decoder_cells,
                                   dropout)
    
    model.eval()
    
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

    print("Val example")
    get_results(model, vocab, val_image, val_caption, val_file)

    # is it overfitting? Let's look at train
    evaluateRandomly(model, train_loader, vocab, n=10, print_data=True)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, 
                        default="models/pytorch_transformer_model_30_epochs.pt", 
                        required=False, 
                        help="path to saved model checkpoint")
    parser.add_argument("--dataset_dir", type=str, 
                        default="flickr8k", 
                        help="path to flickr8k folder")
    opt = parser.parse_args()

    main(opt)