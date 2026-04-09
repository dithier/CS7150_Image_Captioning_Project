from old_transformer_files.transformer_model import VisionTransformerModel
import torch
from dataloader import get_flickr8k_loaders

# P = 8
# num_heads = 8
# ff_dim = 512
# num_encoder_cells = 8
# num_decoder_cells = 8

P = 16
embed_dim = 256 # hidden size D in vision transformer?
num_heads = 8 # ViT Base
trx_ff_dim = embed_dim * 4  # couldn't find in paper
num_encoder_cells = 6 # ViT Base
num_decoder_cells = 6 # ViT Base
dropout = 0.1

train_loader, val_loader, test_loader, vocab = get_flickr8k_loaders("flickr8k")

for i, batch_data in enumerate(train_loader):
    # Every data instance is an image + label pair
    images, labels, _ = batch_data
    break

model = VisionTransformerModel(vocab, P, embed_dim, num_heads, trx_ff_dim,
                                   num_encoder_cells, num_decoder_cells,
                                   dropout)

# x = torch.randn(32, 3, 32, 32) # B X C X H X W
 
# labels is B X L
output = model(images, labels)

english = model.generate(images)


