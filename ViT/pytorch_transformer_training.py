import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
from dataloader_v2 import get_flickr8k_loaders
from training_helpers import *
from pytorch_transformer_enc_dec_model import VisionTransformerModel
import os

# pip install tensorboard
# pip install pandas
# pip install nltk

"""
More correct version of baseline_training.py.
Key differences from diy_transformer_training.py:
  - using inference loss for saving model and evaluating
  - additional prints on random batches to see if captioning makes sense

To run locally:
  python diy_transformer_training_2.py --epochs 30 --checkpoint False --dataset_dir flickr8k
      --log_dir runs/adam_pass_1 --save_path saved_models/adam_pass_1/

To run on cluster:
  Use diy_transformer_training.sh (same structure as baseline_train.sh)
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Logger- Redirect stdout to a log file while still printing to console
# class Logger:
#     def __init__(self, filepath):
#         self.terminal = sys.__stdout__
#         self.log = open(filepath, "w")
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#         self.log.flush()
#     def flush(self):
#         self.terminal.flush()
#         self.log.flush()

# sys.stdout = Logger(os.path.join("logs", "adam_pass_3.out"))

############## Checkpoint Related Logic #############################

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, lr, lr_sched,
                    best_perf, weight_decay, path="./model.pt"):
    torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'best_perf': best_perf,
          'lr': lr,
          'lr_sched': lr_sched.state_dict(),
          'train_loss': train_loss,
          'val_loss': val_loss,
          'weight_decay': weight_decay,   # needed to restore Adam correctly
          }, path)
    print("Saved checkpoint to", path)

def load_checkpoint(model, mode, path="./model.pt"):
    checkpoint = torch.load(path)
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


############## Training Related Logic ####################################

def train_val_model(opt, vocab, model, train_data_loader, val_data_loader, loss_fn,
                    optimizer, lr_scheduler, current_lr, curr_epoch, total_epochs,
                    best_perf, print_save_freq):

    writer = SummaryWriter(opt.log_dir)

    last_lr = current_lr

    for epoch_i in range(curr_epoch, total_epochs + 1):
        model.train()

        running_loss = 0.0
        running_total = 0.0
        batches_since_last_log = 0

        for i, batch_data in enumerate(train_data_loader):
            images, captions, _ = batch_data
            images = images.to(device)
            captions = captions.to(device)

            optimizer.zero_grad()

            # todo captions[:, :-1]?
            outputs = model(images, captions[:, :-1]) # this is train so uses causal mask

            # todo captions[:, 1:]? (address in training helpers too)
            loss = loss_fn(
                outputs.reshape(-1, len(vocab)),
                captions[:, 1:].reshape(-1) # we don't want to include SOS
            )

            loss.backward()

            # todo: need this?
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            running_loss += loss.item()
            running_total += captions.size(0)
            batches_since_last_log += 1

            if i > 0 and i % print_save_freq == 0 or i == len(train_data_loader) - 1:

                # avg loss across batches
                running_loss = running_loss / batches_since_last_log

                avg_val_loss = get_avg_validation_transformer_loss(model, val_data_loader, loss_fn, vocab)

                writer.add_scalars("Loss", {
                    "train": running_loss,
                    "val": avg_val_loss
                }, epoch_i * len(train_data_loader) + i)

                try:
                    last_lr = lr_scheduler.get_last_lr()[0]
                except AttributeError:
                    pass

                if avg_val_loss < best_perf:
                    best_perf = avg_val_loss
                    save_path_labeled = os.path.join(opt.save_path, "model.pt")
                    save_checkpoint(model, optimizer, epoch_i, running_loss, avg_val_loss,
                                    last_lr, lr_scheduler, best_perf, opt.weight_decay,
                                    save_path_labeled)
                
                if epoch_i == total_epochs and i == len(train_data_loader) - 1:
                    save_path_labeled = os.path.join(opt.save_path, f"model_{epoch_i}_epochs.pt")
                    save_checkpoint(model, optimizer, epoch_i, running_loss, avg_val_loss,
                                    last_lr, lr_scheduler, best_perf, opt.weight_decay,
                                    save_path_labeled)

                print(f'[{epoch_i}/{total_epochs}, {i + 1:5d}/{len(train_data_loader)}] avg train loss: {running_loss:.3f} avg val loss: {avg_val_loss:.3f} lr: {last_lr:.6f}')
                running_loss = 0.0
                running_total = 0.0
                batches_since_last_log = 0

        lr_scheduler.step()


def main(opt):
    train_loader, val_loader, _, vocab = get_flickr8k_loaders(root_dir=opt.dataset_dir)

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

    tmax = opt.epochs + 1

    if opt.checkpoint:
        model, optimizer_state_dict, curr_epoch, lr, lr_sched_state, best_perf, weight_decay = load_checkpoint(model, "train", opt.checkpoint_path)

        loss_fn, optimizer = set_up_Adam_loss_optimizer(model, lr, opt.betas, weight_decay, vocab)
        optimizer.load_state_dict(optimizer_state_dict)

        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1)
        # lr_scheduler = set_up_cos_annealing_lr_scheduler(optimizer, tmax)
        lr_scheduler.load_state_dict(lr_sched_state)

        curr_lr = lr
    else:
        loss_fn, optimizer = set_up_Adam_loss_optimizer(model, opt.lr, opt.betas, opt.weight_decay, vocab)
        curr_epoch = 1
        lr_scheduler = set_up_cos_annealing_lr_scheduler(optimizer, tmax)
        best_perf = float("inf")
        curr_lr = opt.lr

    train_val_model(opt, vocab, model, train_loader, val_loader, loss_fn,
                    optimizer, lr_scheduler, curr_lr, curr_epoch, opt.epochs, best_perf, 
                    print_save_freq=opt.print_freq)


############## Helper Fns for Args ############################
def str2bool(arg):
    if isinstance(arg, bool):
        return arg
    if arg == "True":
        return True
    elif arg == "False":
        return False
    else:
        raise argparse.ArgumentTypeError("boolean expected")

def parse_betas(s):
    # accepts "(0.9,0.999)" or "0.9,0.999"
    s = s.strip("() ")
    parts = s.split(",")
    return (float(parts[0]), float(parts[1]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--print_freq", type=int, default=50, help="how many batches go by in between save and prints")
    parser.add_argument("--checkpoint", type=str2bool, default=True, help="true to load from checkpoint, false otherwise")
    parser.add_argument("--checkpoint_path", type=str, default="./model.pt", help="path to checkpoint")
    parser.add_argument("--dataset_dir", type=str, default=".", help="directory for all images/annotations")
    parser.add_argument("--lr", type=float, default=1e-3, help="starting learning rate")
    parser.add_argument("--betas", type=parse_betas, default="0.9,0.999", help="Adam betas as 'b1,b2'")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Adam weight decay")
    parser.add_argument("--log_dir", type=str, default="./runs/diy_transformer_pass_1", help="tensorboard log directory")
    parser.add_argument("--save_path", type=str, default="./saved_models/diy_transformer_pass_1/", help="directory to save checkpoints")

    opt = parser.parse_args() 

    main(opt)