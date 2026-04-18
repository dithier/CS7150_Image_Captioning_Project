
from dataloader_v2 import get_flickr8k_loaders
from training_helpers import *
#from ViT.transformer_enc_doc_model import VisionTransformerModel
from ViT.pytorch_transformer_enc_dec_model import VisionTransformerModel
from baseline.baseline_model_v2 import BaselineModel

def load_checkpoint(model, mode, path="./model.pt"):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    epoch = checkpoint['epoch']
    lr = checkpoint['lr']
    weight_decay = checkpoint['weight_decay']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    lr_sched = checkpoint['lr_sched']
    best_perf = checkpoint['best_perf']

    if mode == "eval":
        model.eval()
    else:
        model.train()

    return model, optimizer_state_dict, epoch, lr, lr_sched, best_perf, weight_decay

train_loader, val_loader, test_loader, vocab = get_flickr8k_loaders(root_dir="flickr8k")

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

model, _, _, _, _, _, _ = load_checkpoint(model, "eval", "models/overfit_model_30_epochs.pt")

model.train()
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"total param: {total_params}")
print(f"trainable param: {trainable_params}")


model.eval()
evaluateRandomly(model, val_loader, vocab, n=10)


# model.train()
# for batch_data in train_loader:
#     images, captions, _ = batch_data

#     # todo captions[:, :-1]?
#     # outputs = model(images, captions[:, :-1])
#     model = BaselineModel(vocab_size=len(vocab))

#     break 

# evaluateRandomly(model, val_loader, vocab)

# loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.word_to_index[vocab.PAD_TOKEN])

# get_avg_validation_transformer_loss(model, val_loader, loss_fn, vocab)

# run two very different images through encoder. if diff near 0 then nearly identical feats fom encoder
# want it to be around > 0.1
# img1_features = encoder_output[0]  # first image in batch
# img2_features = encoder_output[1]  # second image in batch

# diff = (img1_features - img2_features).abs().mean()
# print(f"Mean absolute difference between two images: {diff.item():.4f}")