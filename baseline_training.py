import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
from Dataloader import get_flickr8k_loaders
from training_helpers import *
from models import BaselineModel
from eval_metrics import evaluation_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############## Checkpoint Related Logic #############################

# if we switch to Adam we'll need to save weight decay
# but for now, baseline is SGD + momentum
def save_checkpoint(model, optimizer, epoch, loss, lr, lr_sched, 
                    best_perf, path="./model.pt"):
    torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'best_perf': best_perf,
          'lr': lr,
          'lr_sched': lr_sched,
          'loss': loss,
          }, path)
    print("Saved checkpoint to", path)

# mode should be train or eval
def load_checkpoint(model, mode, path="./model.pt"):
      checkpoint = torch.load(path)
      epoch = checkpoint['epoch']
      lr = checkpoint['lr']
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer_state_dict = checkpoint['optimizer_state_dict']
      loss = checkpoint['loss']
      lr_sched = checkpoint['lr_sched']
      best_perf = checkpoint['best_perf']

      # weightdecay used in adam but for baseline we are using SGD + momentum
      #  weight_decay = checkpoint['weight_decay']
    
      if mode == "eval":
        model.eval()
      else:
        model.train()
    
      return model, optimizer_state_dict, epoch, loss, lr, lr_sched, best_perf #weight_decay,


############## Training Related Logic ####################################

def train_val_model(opt, model, train_data_loader, val_data_loader, loss_fn, 
                    optimizer, lr_scheduler, curr_epoch, total_epochs,
                      best_perf=0, print_freq=50,
                    save_freq=50):
    """
    Training and validating a model using PyTorch.

    Inputs:
      - model: A model implemented in PyTorch
      - data_loader: A data loader that will provide batched images and captions
      - loss_fn: A loss function (e.g., cross entropy loss)
      - lr_scheduler: Learning rate scheduler
      - num_epochs: Number of epochs in total
      - print_freq: Frequency to print training statistics

    Output:
      - model: Trained CNN model
    """

    writer = SummaryWriter(opt.log_dir)

    # TODO: better naming for variables of "running_metric" once we choose one
    for epoch_i in range(curr_epoch, total_epochs + 1):
        # set the model in the train mode so the batch norm layers will behave correctly
        model.train()

        running_loss = 0.0
        running_total = 0.0
        running_metric = 0.0
        for i, batch_data in enumerate(train_data_loader):
            # Every data instance is an image + label pair
            images, labels = batch_data
            images = images.to_device(device)
            labels = labels.to_device(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # use the model
            # TODO: Next few lines are going to be a little different than what was copied
            #  from hw
            outputs = model(images)

            #print(outputs)
            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_total += labels.size(0)
            
            # eval metric
            running_metric += evaluation_metric(labels, outputs)

            # save at regular intervals as well as at end of epoch
            if i % save_freq or i == len(train_data_loader):
                running_acc = running_metric / running_total * 100

                if running_acc > best_perf:
                    best_perf = running_acc
                    last_lr = lr_scheduler.get_last_lr()[0]

                    save_path = opt.save_path.split(".")
                    save_path_labeled = "".join(save_path[:-1])
                    save_path_labeled += f"_epoch_{epoch_i}_iter_{i}." + save_path[-1]

                    save_checkpoint(model, optimizer, epoch_i, loss, 
                                    last_lr, lr_scheduler, best_perf, save_path_labeled)

            if i % print_freq == 0:    # print and log every certain number of mini-batches
                running_loss = running_loss / print_freq
    
                running_metric = running_metric / running_total * 100
                
                writer.add_scalar("Loss/train", running_loss,
                                  epoch_i*(batch_data[0].shape[0] + i))

                # switch out metric
                print(f'[{epoch_i + 1}/{total_epochs}, {i + 1:5d}/{len(train_data_loader)}] loss: {running_loss:.3f} metric: {running_metric:.3f} lr: {last_lr:.5f}')
                running_loss = 0.0
                running_total = 0.0
                running_metric = 0.0
            
        # adjust the learning rate
        lr_scheduler.step()

        val_acc = test_model(model, val_data_loader)
        # switch out metric
        print(f'[{epoch_i + 1}/{total_epochs}] val acc: {val_acc:.3f}')

    return model

def main(opt):
    # load dataset
    train_loader, val_loader, test_loader, vocab = get_flickr8k_loaders(root_dir=opt.dataset_dir)

    # make model and load from checkpoint if needed
    model = BaselineModel()

    ##### set up params
    
    # are we loading from a checkpoint?
    if opt.checkpoint:
        model, optimizer_state_dict, curr_epoch, lr, lr_scheduler, best_perf = load_checkpoint(model, "train", opt.checkpoint_path)                                                                                              path=opt.checkpoint_path)
        loss_fn, optimizer = set_up_SGD_loss_optimizer(model, lr, opt.momentum)
        optimizer.load_state_dict(optimizer_state_dict)
    else:
        loss_fn, optimizer = set_up_SGD_loss_optimizer(model, opt.lr, opt.momentum)
        curr_epoch = 0
        # this is for cosine annealing lr scheduler. This is a variable we can change
        tmax = 10
        lr_scheduler = set_up_cos_annealing_lr_scheduler(optimizer, tmax)
        best_perf


    # call train_val_model
    train_val_model(opt, model, train_loader, val_loader, loss_fn, 
                    optimizer, lr_scheduler, curr_epoch, opt.epochs, best_perf, print_freq=50,
                    save_freq=50)


############## Helper Fns for Args ############################
def str2bool(arg):
    if isinstance(arg, bool):
        return argparse
    if arg == "True":
        return True
    elif arg == "False":
        return False
    else:
        raise argparse.ArgumentTypeError("boolean expected")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--checkpoint", type=str2bool, default=True, help="true to load from checkpoint, false otherwise")
    parser.add_argument("--checkpoint_path", type=str, default="./model.pt", help="path pointing towards checkpoint to load")
    parser.add_argument("--dataset_dir", type=str, default=".", help="directory for all images/annotations")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum float for SGD")
    parser.add_argument("--lr", type=float, default=0.001, help="starting learning rate")
    parser.add_argument("--log_dir", type=str, default="./runs", help="directory that tensorboard results should be logged to")
    parser.add_argument("--save_path", type=str, default=".", help="directory models and checkpoints should be saved in")
    opt = parser.parse_args()

    main(opt)
