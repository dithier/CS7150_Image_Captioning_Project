import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.utils.data import Subset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This is only for LSTM model 
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

def evaluate(model, image, vocab):
    model.eval()
    with torch.no_grad():
        outputs = model(image)

        _, topi = outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == vocab.word_to_index["<EOS>"]:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(vocab.index_to_word[idx.item()])
    return decoded_words

def evaluateRandomly(model, val_data_loader, vocab, n=5):
    indices = torch.randperm(len(val_data_loader))[:n]
    subset_dataset = Subset(val_data_loader.dataset, indices)
    subset_loader = DataLoader(subset_dataset, batch_size=1)
    
    for image, caption, _ in subset_loader:
        image = image.to(device)
        caption = caption.to(device)
        
        output_words = evaluate(model, image, vocab)
        output_sentence = ' '.join(output_words)

        truth = vocab.decode(caption.squeeze().tolist())
        print(f"truth == {truth}")
        print('output >', output_sentence)
        print('')

def get_avg_validation_transformer_loss(model, val_data_loader, loss_fn, vocab):
    running_loss = 0

    model.eval()
    with torch.no_grad():
        for i, batch_data in enumerate(val_data_loader):
            # Every data instance is an image + label pair
            images, captions, _ = batch_data
            images = images.to(device)
            captions = captions.to(device)

            # use the model
            # todo: need to delete last token?
            outputs = model(images) 

            loss = loss_fn(
                outputs.reshape(-1, len(vocab)), # first dimension is batch * seq length, second dim vocab size
                captions[:, 1:].reshape(-1) # groud truth, ignore <SOS> tag
            )

            running_loss += loss.item()
    
    evaluateRandomly(model, val_data_loader, vocab)

    model.train()
    
    return running_loss / len(val_data_loader)


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

def set_up_cos_annealing_warm_restarts_scheduler(optimizer, t_0, t_mult=1):
    return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=t_mult)