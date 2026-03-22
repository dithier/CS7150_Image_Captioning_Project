import torch.nn as nn
import torchvision.models as models
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Baseline model:
Pretrained RESNET encoder
LSTM decoder (without attention)
"""
class ResNetEncoder(nn.Module):
    def __init__(self, embed_dim=512, freeze=True):
        # 512 was what was used in Vinyal's show and tell
        # freeze is a parameter of whether or not we want to freeze resnet layers during
        # training
        super().__init__()
        resnet = models.resnet50(weights="IMAGENET1K_V1")

        # we don't want the classification layer (last layer) for resnet bc we are not doing
        # classification. We want to take the features learned before that layer
        # and feed it into LSTM
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # We want to take resnet prev layer output (2048) and
        # make it the embed dim size for the LSM input
        self.fc = nn.Linear(2048, embed_dim)

        if freeze:
            # freeze encoder for baseline
            for p in self.resnet.parameters():
                p.requires_grad = False

    def forward(self, x):
        features = self.resnet(x) # (B, 2048, 1, 1)
        features = features.flatten(1) # needed bc last layer was global avg pooling (B, 2048)
        return self.fc(features) # at this point we have (B, embed_dim)

class BasicLSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=512, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.init_h = nn.Linear(embed_dim, hidden_dim) # this gives our h0 state from the image to go into LSTM
        self.init_c = nn.Linear(embed_dim, hidden_dim)

    def forward(self, features, captions):
        # the unsqueeze gives us (1, B, hidden_dim)
        h0 = self.init_h(features).unsqueeze(0) 
        c0 = self.init_c(features).unsqueeze(0)

        embeds, _ = self.lstm(self.embed(captions), (h0, c0))
        return self.fc(embeds)
    
    # this returns in the INDEXES of the words in the covab
    def generate(self, features, vocab, max_length=30):
        self.eval()

        B = features.size(0) # batch num

        # initialize hidden state of LSTM just like beginning of forward
        h0 = self.init_h(features).unsqueeze(0) 
        c0 = self.init_c(features).unsqueeze(0)

        token = torch.full(
            (B, 1), # make a tensor for dimensions batch size, 1
            vocab.word_to_index[vocab.SOS_TOKEN], # make every row start with the Start of Sentence (SOS) token
            dtype=torch.long, device=device
        )

        # this has a emty list for each batch image. These lists will be appended to as we generate 
        # the caption below
        captions   = [[] for _ in range(B)]

        # 1 means we are finished with that image in the batch, 0 otherwise
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        h, c = h0, c0

        with torch.no_grad():
            for _ in range(max_length):
                embed = self.embed(token) # (B , 1, embed_dim)
                out, (h,c) = self.lstm(embed, (h,c)) # out: (B, 1, hidden_dim)
                logits = self.fc(out.squeeze(1)) # (B, vocab size)
                predicted = logits.argmax(dim=-1) # (B, )  this is the predicted index of the word for this time step
            
                # we now have a prediction for each batch, lets add them to our caption list            
                for i in range(B):
                    if not finished[i]:
                        # not finished? get the predicted word
                        idx = predicted[i].item()
                        # if that word is end of sentence token, mark as done
                        if idx == vocab.word_to_index[vocab.EOS_TOKEN]:
                            finished[i] = True
                        else:
                            # otherwise, apply word to caption
                            captions[i].append(vocab.index_to_word[idx])

                # everything finished? we can break, we're done
                if finished.all():
                    break

                # still got more to go? make the token whte write dimension to continue loop
                token = predicted.unsqueeze(1)  # (B, 1) — feed prediction back in

        return captions
    

class BaselineModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, freeze=True, hidden_dim=512,  num_layers=1):
        super().__init__()
        self.encoder = ResNetEncoder(embed_dim, freeze)
        self.decoder = BasicLSTMDecoder(vocab_size, embed_dim, hidden_dim, num_layers)

    def forward(self, images, captions):
        x = self.encoder(images)
        return self.decoder(x, captions)
    
    def generate(self, images, vocab, max_length=30):
        features = self.encoder(images)
        return self.decoder.generate(features, vocab, max_length)

