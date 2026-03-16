import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from eval_metrics import evaluation_metric, prepare_hypotheses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_avg_validation_loss(model, val_data_loader, loss_fn, vocab):
    running_loss = 0

    model.eval()
    with torch.no_grad():
        for i, batch_data in enumerate(val_data_loader):
            # Every data instance is an image + label pair
            images, captions, _ = batch_data
            images = images.to(device)
            captions = captions.to(device)

            # use the model
            outputs = model(images, captions[:, :-1]) # we want <SOS> to last word but not <EOS> token

            loss = loss_fn(
                outputs.reshape(-1, len(vocab)), # first dimension is batch * seq length, second dim vocab size
                captions[:, 1:].reshape(-1) # groud truth, ignore <SOS> tag
            )

            running_loss += loss.item()
    
    model.train()
    
    return running_loss / len(val_data_loader)

# TODO (priyanshu) need generate function, behaves differently than training
# this probably shouldn't be accuracy, maybe BLEU-4?
# if we use BLEU-4 it will need to be across the whole dataset, not just batches
# also, better results if we get ALL captions for image (ex flickr8 has 5 per image)
"""
def test_model(model, data_loader, vocab):
    
    Compute (BLUE 4?) performance of the model.

    Inputs:
      - model: A encoder/decoder model  implemented in PyTorch
      - data_loader: A data loader that will provide batched images and targets
    

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

"""

def test_model(model, data_loader, vocab):
    model.eval()

    all_generated = []

    with torch.no_grad():
        for batch_data in data_loader:
            images, captions, _ = batch_data
            images = images.to(device)

            batch_generated = model.generate(images, vocab)
            all_generated.extend(batch_generated)

    hypotheses = prepare_hypotheses(all_generated, vocab)
    bleu_scores = evaluation_metric(data_loader, hypotheses)

    return bleu_scores

def set_up_SGD_loss_optimizer(model, learning_rate, momentum, vocab):
    """
    In this programming assignment, we will adopt the most common choice for the optimizer:
    SGD + momentum and learning rate scheduler: StepLR. Please refer to https://pytorch.org/docs/stable/optim.html#algorithms
    and https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR for more details.
    """
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.word_to_index[vocab.PAD_TOKEN]) # needed to ignore index 0 whenever it shows up which is definited as PAD in our vocab
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    return loss_fn, optimizer

def set_up_Adam_loss_optimizer(model, learning_rate, betas, weight_decay, vocab):
    """
    In this programming assignment, we will adopt the most common choice for the optimizer:
    SGD + momentum and learning rate scheduler: StepLR. Please refer to https://pytorch.org/docs/stable/optim.html#algorithms
    and https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR for more details.
    """
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.word_to_index[vocab.PAD_TOKEN]) # needed to ignore index 0 whenever it shows up which is defined as PAD in our vocab
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas,
                           weight_decay=weight_decay)
    
    return loss_fn, optimizer

def set_up_step_lr_scheduler(optimizer, lr_step_size, lr_gamma):
    return optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

def set_up_cos_annealing_lr_scheduler(optimizer, t_max):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max)