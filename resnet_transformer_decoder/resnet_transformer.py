import torch.nn as nn
import torchvision.models as models
import torch
from ViT.transformer_enc_doc_model import PositionalEncoding
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Variation of baseline
Pretrained RESNET encoder
Transformer decoder
"""
class ResNetEncoder(nn.Module):
    def __init__(self, embed_dim=512, freeze=True):
        # 512 was what was used in Vinyal's show and tell
        # freeze is a parameter of whether or not we want to freeze resnet layers during
        # training
        super().__init__()
        resnet = models.resnet50(weights="IMAGENET1K_V1")

        # we don't want the classification layer (last layer) for resnet bc we are not doing
        # classification. We also don't want the avg pool layer because we want to keep the spatial feature maps
        # so transformer decoder can cross-attend
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules) 

        # We want to take resnet prev layer output (2048) and
        # make it the embed dim size for the LSM input
        self.fc = nn.Linear(2048, embed_dim)

        if freeze:
            # freeze encoder for baseline
            for p in self.resnet.parameters():
                p.requires_grad = False

    # todo: we need (B, L, embed_dim) -> remove avgpool res
    def forward(self, x):
        features = self.resnet(x) # (B, 2048, 7, 7)

        B, C, H, W = features.size()
        features = features.permute(0, 2, 3, 1) # (B, 7, 7, 2048)
        features = features.reshape(B, H * W, C)

        return self.fc(features) # at this point we have (B, 49, embed_dim)

class TranformerDecoder(nn.Module):
    def __init__(self, vocab, embed_dim, num_heads, trx_ff_dim: int=3072, num_decoder_cells : int=6,
              dropout: float=0.1):
        super().__init__()
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.pad_token = vocab.word_to_index["<PAD>"]

        self.positional_encoding = PositionalEncoding(self.embed_dim)
        self.embedding = nn.Embedding(len(vocab), self.embed_dim)

        layer_norm_2 = nn.LayerNorm(self.embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(self.embed_dim, num_heads, dim_feedforward=trx_ff_dim, 
                                                   dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_cells, norm=layer_norm_2) 
        
        self.fc_out = nn.Linear(self.embed_dim, len(vocab))

    def forward_train(self, encoder_output, labels):
        """
        Inputs:
        - image: is input image(s) of shape B x C X H X W
        - labels: Tensor with the shape of BxL, containing the indexes of each word in
          the vocabulary, which will be converted into word embeddings with the shape
          of BxLxC
        - mask: Tensor for masking in the multi-head attention

        Return:
        - logits: Tensor with the shape of BxK, where K is the number of classes
        """
        # labels is B x seq_len
        seq_len = labels.size(1)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(labels.device) #seq len X seq len

        padding_mask = (labels == self.pad_token) # B x seq len

        logits = None
        ###########################################################################
        # Apply positional embedding to the input, which is then fed into   #
        # the encoder. Average pooling is applied then to all the features of all #
        # tokens. Finally, the logits are computed based on the pooled features.  #
        ###########################################################################
        # label embeddings B X L
        label_embeddings = self.embedding(labels) * math.sqrt(self.embed_dim) # B X L X C
        # add positional encoding
        label_embeddings = self.positional_encoding(label_embeddings)

        # encoder_output is B x 49 x embed_dim
        decoder_output = self.transformer_decoder(label_embeddings, encoder_output, tgt_mask=causal_mask,    
                                                  tgt_key_padding_mask=padding_mask,
                                                  memory_key_padding_mask=None,
                                                  tgt_is_causal=True) 
        
        logits = self.fc_out(decoder_output) # B X L X vocab_size

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return logits
    
    def forward_test(self, encoder_output, max_length=30):
        # batch num
        B = encoder_output.shape[0]

        # start with just <SOS>
        generated = torch.full(
            (B, 1), # make a tensor for dimensions batch size, 1
            self.vocab.word_to_index[self.vocab.SOS_TOKEN], # make every row start with the Start of Sentence (SOS) token
            dtype=torch.long, device=encoder_output.device
        )
        
        decoder_outputs = []

        for _ in range(max_length + 1):
            decoder_input = self.positional_encoding(self.embedding(generated) * math.sqrt(self.embed_dim))
            decoder_output = self.transformer_decoder(decoder_input, encoder_output)

            output = self.fc_out(decoder_output)
            last_output = output[:, -1, :]

            decoder_outputs.append(last_output)

            # get indices of highest values 
            _, topi = last_output.topk(1)
        
            generated = torch.cat((generated, topi), dim=1)
        
        logits = torch.stack(decoder_outputs, dim=1)

        return logits


class ResnetTransformerModel(nn.Module):
    def __init__(self, vocab, num_heads, trx_ff_dim, num_decoder_cells, embed_dim=512, dropout= 0.1, freeze=True):
        super().__init__()
        self.vocab = vocab
        self.embed_dim = embed_dim

        self.encoder = ResNetEncoder(embed_dim, freeze)

        self.decoder = TranformerDecoder(self.vocab, self.embed_dim, num_heads, trx_ff_dim, num_decoder_cells, dropout)
    
    def forward_train(self, images, captions):
        encoder_output = self.encoder(images)
        return self.decoder.forward_train(encoder_output, captions)

    def forward_test(self, images):
        encoder_output = self.encoder(images)
        return self.decoder.forward_test(encoder_output)

    # returns logits
    def forward(self, images, labels=None):
        """
        Inputs:
        - image: is input image(s) of shape B x C X H X W
        - labels: Tensor of BxLd, word indexes of the english caption. Can be empty.

        Return:
        - y: Tensor of BxLdxK, corresponding to the log probabilities of generated
             words in the target language. K is the vocab size.
        """

        if self.training:
            # training-time behavior
            assert labels is not None
            return self.forward_train(images, labels)

        # testing-time behavior
        return self.forward_test(images)
    

