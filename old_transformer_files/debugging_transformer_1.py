import torch
from dataloader_v2 import get_flickr8k_loaders
from training_helpers import *
from old_transformer_files.transformer_model import VisionTransformerModel


device = "gpu" if torch.cuda.is_available() else "cpu"

def load_checkpoint(model, mode, path="./model.pt"):
    checkpoint = torch.load(path, map_location="cpu")
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
                                dropout).to(device)

model, optimizer_state_dict, curr_epoch, lr, lr_sched_state, best_perf, weight_decay = load_checkpoint(model, "eval", "transformer_decoder/model.pt")

loss_fn, optimizer = set_up_Adam_loss_optimizer(model, lr, (0.9, 0.999), weight_decay, vocab)
optimizer.load_state_dict(optimizer_state_dict)

model.eval()

with torch.no_grad():
    for i, batch_data in enumerate(val_loader):
            # Every data instance is an image + label pair
        images, captions, _ = batch_data
        images = images.to(device)
        captions = captions.to(device)

        # use the model
        outputs = model(images, captions[:, :-1]) # we want <SOS> to last word but not <EOS> token

        probs = torch.softmax(outputs[:, -1, :], dim=-1)
        top_prob, top_idx = probs.topk(1)

        print(f"top prob: {top_prob}\n" \
                "idx:  {top_idx}\n")

        loss = loss_fn(
            outputs.reshape(-1, len(vocab)), # first dimension is batch * seq length, second dim vocab size
            captions[:, 1:].reshape(-1) # groud truth, ignore <SOS> tag
        )

        print(f"loss {loss}")
        break

with torch.no_grad():
    for i, batch_data in enumerate(val_loader):
            # Every data instance is an image + label pair
        images, captions, _ = batch_data
        images = images.to(device)
        captions = captions.to(device)

        B = images.size(0)

        # start with just <SOS>
        generated = torch.full(
            (B, 1), # make a tensor for dimensions batch size, 1
            vocab.word_to_index[vocab.SOS_TOKEN], # make every row start with the Start of Sentence (SOS) token
            dtype=torch.long, device=images.device
        )

         # use the model
        outputs = model(images, generated) # we want <SOS> to last word but not <EOS> token

        probs = torch.softmax(outputs[:, -1, :], dim=-1)
        top_prob, top_idx = probs.topk(1)

        print(f"top prob: {top_prob}\n" \
                "idx:  {top_idx}\n")
        
        break

        