import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from eval_metrics import evaluation_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO (priyanshu) need generate function, behaves differently than training
# this probably shouldn't be accuracy, maybe BLEU-4?
# if we use BLEU-4 it will need to be across the whole dataset, not just batches
# also, better results if we get ALL captions for image (ex flickr8 has 5 per image)
def test_model(model, data_loader, vocab):
    """
    Compute (BLUE 4?) performance of the model.

    Inputs:
      - model: A encoder/decoder model  implemented in PyTorch
      - data_loader: A data loader that will provide batched images and targets
    """

    # set model to eval mode
    model.eval()

    # TODO: we prob want one of our metrics like BLEU. Once decided, fix logic and names
    metric = 0
    total = 0

    # we are not training, we are evaluating here, so we don't need to calculate
    # gradients for our outputs
    with torch.no_grad():
        for batch_data in data_loader:
            images, targets = batch_data
            images = images.to(device)
            targets = targets.to(device)

            predicted_caption = model.generate(images, vocab)

            total += targets.size(0)
            metric += evaluation_metric(targets, predicted_caption)
    
    metric = 100 * metric // total
    return metric

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

def set_up_cos_annealing_lr_scheduler(optimizer, t_max):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max)