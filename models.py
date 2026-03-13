import torch.nn as nn
import torchvision.models as models
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Baseline model:
Pretrained ResNet50 encoder (SAT-style, outputs spatial feature maps)
LSTM decoder (without attention)

SAT Encoder explanation:
    The original baseline encoder collapsed the image into a single vector
    (B, embed_dim) by keeping avgpool. This loses all spatial information —
    the decoder has no idea WHERE in the image things are.

    For Show, Attend and Tell (SAT), the encoder must preserve spatial
    structure so the attention mechanism can focus on different regions
    when generating each word.

    We do this by removing BOTH avgpool and the final classification layer
    (the last 2 layers of ResNet), keeping the convolutional layers only.
    This gives us a 14x14 grid of feature vectors — one per image region.

    Input  : (B, 3, 224, 224)
    Output : (B, 196, embed_dim)
             196 = 14x14 spatial locations
             Each location has embed_dim features describing that region
"""
class ResNetEncoder(nn.Module):
    def __init__(self, embed_dim=512, freeze=True):
        """
        Parameters
        ----------
        embed_dim : int    size of the feature vector for each spatial location.
                           512 was used in Vinyals' Show and Tell.
        freeze    : bool   if True, freeze all ResNet weights during training.
                           Set to False in later phases to fine-tune the encoder.
        """
        super().__init__()
        resnet = models.resnet50(weights="IMAGENET1K_V1")

        # Remove the last 2 layers (avgpool + fc classification layer).
        # Keeping only convolutional layers preserves the 14x14 spatial grid.
        # Compare to baseline which only removed the last 1 layer (fc),
        # keeping avgpool which collapsed spatial info to (B, 2048, 1, 1).
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Ensure spatial size is exactly 14x14 regardless of input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))

        # Project each of the 2048-dimensional spatial features to embed_dim.
        # Applied independently to each of the 196 spatial locations.
        self.fc = nn.Linear(2048, embed_dim)

        if freeze:
            # Freeze all ResNet layers — only the projection fc will be trained.
            # In Phase 4 experiments, set freeze=False to fine-tune ResNet too.
            for p in self.resnet.parameters():
                p.requires_grad = False

    def forward(self, x):
        # Pass through frozen ResNet convolutional layers
        features = self.resnet(x)               # (B, 2048, H, W)

        # Ensure spatial size is exactly 14x14 regardless of input size
        features = self.adaptive_pool(features) # (B, 2048, 14, 14)

        # Reshape: move spatial dimensions into the sequence dimension
        # so each of the 196 locations becomes one "token" for the decoder
        B, C, H, W = features.size()
        features = features.permute(0, 2, 3, 1)  # (B, 14, 14, 2048)
        features = features.view(B, H * W, C)     # (B, 196, 2048)

        # Project each spatial location from 2048 -> embed_dim
        features = self.fc(features)              # (B, 196, embed_dim)

        return features

class BasicLSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=512, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.init_h = nn.Linear(embed_dim, hidden_dim) # this gives our h0 state from the image to go into LSTM

    def forward(self, features, captions):
        # the unsqueeze gives us (1, B, hidden_dim)
        h0 = self.init_h(features).unsqueeze(0) 
        c0 = torch.zeros_like(h0)

        embeds, _ = self.lstm(self.embed(captions), (h0, c0))
        return self.fc(embeds)
    
    # this returns in the INDEXES of the words in the covab
    def generate(self, features, vocab, max_length=30):
        self.eval()

        B = features.size(0) # batch num

        # initialize hidden state of LSTM just like beginning of forward
        h0 = self.init_h(features).unsqueeze(0) 
        c0 = torch.zeros_like(h0)

        token = torch.full(
            (B, 1), # make a tensor for dimensions batch size, 1
            vocab.stoi[vocab.SOS_TOKEN], # make every row start with the Start of Sentence (SOS) token
            dtype=torch.long, device=device
        )

        # this has a emty list for each batch image. These lists will be appended to as we generate 
        # the caption below
        captions   = [[] for _ in range(B)]

        # 1 means we are finished with that image in the batch, 0 otherwise
        finished = torch.zeros(B, dtype=torch.bool, device=device)

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
                        if idx == vocab.stoi[vocab.EOS_TOKEN]:
                            finished[i] = True
                        else:
                            # otherwise, apply word to caption
                            captions[i].append(idx)

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
    
    def generate_in_english(self, images, vocab, max_length=30):
        index_captions = self.generate(images, vocab, max_length)
        return [vocab.decode(seq) for seq in index_captions]

"""
Other possibilities:
1) Prertrained ResNet with LSTM with attention
2) Pretrained ResNet with Transformer
3) custom CNN with the best of the above decoders
"""