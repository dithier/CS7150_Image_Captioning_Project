import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: check point logic of actual requirements needs check
############## Checkpoint Related Logic #############################
def save_checkpoint(model, optimizer, epoch, loss, lr, lr_sched, 
                    weight_decay, best_perf, path="./model.pt"):
    torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'best_perf': best_perf,
          'lr': lr,
          'weight_decay': weight_decay,
          'lr_sched': lr_sched,
          'loss': loss,
          }, path)
    print("Saved checkpoint to", path)

# mode should be train or eval
def load_checkpoint(model, mode, path="./model.pt"):
      epoch = checkpoint['epoch']
      lr = checkpoint['lr']
      weight_decay = checkpoint['weight_decay']
      checkpoint = torch.load(path)
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer_state_dict = checkpoint['optimizer_state_dict']
      loss = checkpoint['loss']
      lr_sched = checkpoint['lr_sched']
      best_perf = checkpoint['best_perf']
    
      if mode == "eval":
        model.eval()
      else:
        model.train()
    
      return model, optimizer_state_dict, epoch, loss, lr, weight_decay, lr_sched, best_perf


############## Training Related Logic ####################################

# this probably shouldn't be accuracy, maybe BLEU-4?
def test_model(model, data_loader):
    """
    Compute (accuracy?) performance of the model.

    Inputs:
      - model: A encoder/decoder model  implemented in PyTorch
      - data_loader: A data loader that will provide batched images and targets
    """

    # set model to eval mode
    model.eval()

    # TODO:  we prob don't want correct, we prob want one of our metrics like BLEU
    correct = 0
    total = 0

    # we are not training, we are evaluating here, so we don't need to calculate
    # gradients for our outputs
    with torch.no_grad():
        for batch_data in data_loader:
            images, targets = batch_data
            images.to_device(device)
            targets.to_device(device)

            logits = model(images)
            # TODO: putting this as place holder bc diff than hw
            predicted = 0

            total += targets.size(0)
            # TODO: calculate eval metric based on logits. for know just using correct
            # which is not what we want
            correct += (predicted == targets.sum().item())
    
    # TODO: this is a placeholder
    acc = 100 * correct // total
    return acc

def set_up_SGD_loss_optimizer(model, learning_rate, momentum):
    """
    In this programming assignment, we will adopt the most common choice for the optimizer:
    SGD + momentum and learning rate scheduler: StepLR. Please refer to https://pytorch.org/docs/stable/optim.html#algorithms
    and https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR for more details.
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    return loss_fn, optimizer

def set_up_Adam_loss_optimizer(model, learning_rate, betas, weight_decay):
    """
    In this programming assignment, we will adopt the most common choice for the optimizer:
    SGD + momentum and learning rate scheduler: StepLR. Please refer to https://pytorch.org/docs/stable/optim.html#algorithms
    and https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR for more details.
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas,
                           weight_decay=weight_decay)
    
    return loss_fn, optimizer

def set_up_step_lr_scheduler(optimizer, lr_step_size, lr_gamma):
    return optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

def set_up_cose_annealing_lr_scheduler(optimizer, t_max):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max)

# TODO: add tensorboard logging
def train_val_model(model, train_data_loader, val_data_loader, loss_fn, 
                    optimizer, lr_scheduler, num_epochs, best_perf=0, print_freq=50,
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

    for epoch_i in range(num_epochs):
        # set the model in the train mode so the batch norm layers will behave correctly
        model.train()

        running_loss = 0.0
        running_total = 0.0
        # TODO: running_correct should be replaced with eval method like BLEU
        running_correct = 0.0
        for i, batch_data in enumerate(train_data_loader):
            # Every data instance is an image + label pair
            images, labels = batch_data
            images = images.to_device(device)
            labels = labels.to_device(device)

            predicted = None
            
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

            # we want the class INDEX that has the highest probability
            # TODO: this will be different in our case
            predicted = torch.argmax(outputs, dim=1)

            # print statistics
            running_loss += loss.item()
            running_total += labels.size(0)
            # TODO running correct needs to be replace with our eval (same with printing  immediately following)
            running_correct += (predicted == labels).sum().item()

            # save at regular intervals as well as at end of epoch
            if i % save_freq or i == len(train_data_loader):
                # TODO: replace with our metric
                running_acc = running_correct / running_total * 100

                if running_acc > best_perf:
                    best_perf = running_acc
                    last_lr = lr_scheduler.get_last_lr()[0]

                    save_path = opt.save_path.split(".")
                    save_path_labeled = "".join(save_path[:-1])
                    save_path_labeled += f"_epoch_{epoch_i}_iter_{i}." + save_path[-1]

                    # TODO: do we need weight decay?
                    save_checkpoint(model, optimizer, epoch_i, loss, 
                                    last_lr, weight_decay, best_perf, save_path_labeled)

            if i % print_freq == 0:    # print and log every certain number of mini-batches
                running_loss = running_loss / print_freq
                # TODO change running accuracy to BLEU?
                running_acc = running_correct / running_total * 100
                

                writer.add_scalar("Loss/train", running_loss,
                                  epoch_i*(batch_data[0].shape[0] + i))

                print(f'[{epoch_i + 1}/{num_epochs}, {i + 1:5d}/{len(train_data_loader)}] loss: {running_loss:.3f} acc: {running_acc:.3f} lr: {last_lr:.5f}')
                running_loss = 0.0
                running_total = 0.0
                running_correct = 0.0
            
        # adjust the learning rate
        lr_scheduler.step()

        val_acc = test_model(model, val_data_loader)
        print(f'[{epoch_i + 1}/{num_epochs}] val acc: {val_acc:.3f}')

    return model

# TODO
def main(opt):
    pass

    # load dataset

    # set up params

    # make model and load from checkpoint if needed

    # call train_val_model


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

# TODO: adjust default values and make sure most have defaults
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--checkpoint", type=str2bool, default=True, help="true to load from checkpoint, false otherwise")
    parser.add_argument("--checkpoint_path", type=str, default="./model.pt", help="path pointing towards checkpoint to load")
    parser.add_argument("--img_dir", type=str, help="directory for train images")
    parser.add_argument("--annotations", type=str, help="path to train annotations json file")
    parser.add_argument("--val_dir", type=str, help="directory for validation images")
    parser.add_argument("--val_annotations", type=str, help="path to validation annotations json file")
    parser.add_argument("--log_dir", type=str, default="./runs", help="directory that tensorboard results should be logged to")
    parser.add_argument("--log_freq", type=int, default=4, help="default number of batches that should elapse before tensorboard logging")
    parser.add_argument("--save_freq", type=int, default=400, help="default number of batches that should elapse before model state is save")
    parser.add_argument("--eval_freq", type=int, default=940, help="default number of batches that should elapse before validation loss is logged")
    parser.add_argument("--save_path", type=str, default=".", help="directory models and checkpoints should be saved in")
    parser.add_argument("--train", type=str2bool, default=True, help="True if you want to train false otherwise")
    opt = parser.parse_args()

    main(opt)