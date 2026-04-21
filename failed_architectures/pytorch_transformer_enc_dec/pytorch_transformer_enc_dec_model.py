"""
Authors: Carter Ithier, Priyanshu Ranka  
Course: CS 7150 - Deep Learning
Semester: Spring 2026
Short description:  Transformer encoder-decoder model for image captioning.
"""
import math
from diy_transformer_enc_dec.transformer_enc_dec_model import PositionalEncoding
import torch
import torch.nn as nn

class VisionTransformerModel(nn.Module):
    """
    A Transformer-based image captioning model
    """
    def __init__(self,
            vocab, P: int, embed_dim: int, num_heads: int, trx_ff_dim: int,
            num_encoder_cells: int, num_decoder_cells : int,
              dropout: float=0.1
        ):
        """
        Inputs:
        - vocab: the vocab object generated from training data
        - P: (P,P) is resolution of each image patch
        - num_heads: Number of attention heads in a multi-head attention module
        - trx_ff_dim: The hidden dimension for a feedforward network
        - num_trx_cells: Number of TransformerEncoderCells
        - dropout: Dropout ratio
        """
        super(VisionTransformerModel, self).__init__()

        self.P = P
        self.C = 3 # number of channels
        self.embed_dim = embed_dim

        self.init_embed_dim = P * P * self.C # from vision transformer paper in HW 3

        self.pad_token = vocab.word_to_index["<PAD>"]
        self.vocab = vocab
        
        ###########################################################################
        # Define a module for positional encoding, Transformer encoder, and #
        # a output layer                                                          #
        ###########################################################################
        self.image_embedding_layer = nn.Linear(self.init_embed_dim, self.embed_dim)
        self.positional_encoding = PositionalEncoding(self.embed_dim)
        self.embedding = nn.Embedding(len(vocab), self.embed_dim)

        layer_norm_1 = nn.LayerNorm(self.embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(self.embed_dim, num_heads, dim_feedforward=trx_ff_dim,
                                                   dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_cells, norm=layer_norm_1,
                                                         enable_nested_tensor=False) 

        layer_norm_2 = nn.LayerNorm(self.embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(self.embed_dim, num_heads, dim_feedforward=trx_ff_dim, 
                                                   dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_cells, norm=layer_norm_2) 
        
        self.fc_out = nn.Linear(self.embed_dim, len(vocab))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def make_patches(self, image):
        # image is B x C X H x W
        patches = torch.nn.functional.unfold(image, self.P, stride=self.P) 

        # patches is B X (P * P * C) X N
        patches = torch.transpose(patches, 1, 2)

        # patches is B X N X (P * P * C)
        return patches

    def forward_train(self, image, labels):
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
        # image is B x C X H x W
    
        # make sure we can split up the image correctly
        assert image.shape[2] % self.P == 0
        assert image.shape[3] % self.P == 0

        # number of patches (from paper N = HW/P^2)
        N = (image.shape[2] * image.shape[3]) // (self.P**2)
    
        # batch num
        B = image.shape[0]

        seq_len = labels.size(1)

        embedded = self.make_patches(image) # B X N X (P * P * C)
        embedded = self.image_embedding_layer(embedded) * math.sqrt(self.embed_dim) # B X N X self.embed_dim

        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(labels.device) #seq len X seq len

        padding_mask = (labels == self.pad_token) # B x seq len

        logits = None
        ###########################################################################
        # Apply positional embedding to the input, which is then fed into   #
        # the encoder. Average pooling is applied then to all the features of all #
        # tokens. Finally, the logits are computed based on the pooled features.  #
        ###########################################################################
        # C' = self.embed_dim
        output = self.positional_encoding(embedded) # B X N X C'
        encoder_output = self.transformer_encoder(output) # B X N X C'

        # label embeddings B X L
        label_embeddings = self.embedding(labels) * math.sqrt(self.embed_dim) # B X L X C
        # add positional encoding
        label_embeddings = self.positional_encoding(label_embeddings)

        decoder_output = self.transformer_decoder(label_embeddings, encoder_output,  tgt_mask=causal_mask,    
                                                  tgt_key_padding_mask=padding_mask,
                                                  memory_key_padding_mask=None,
                                                  tgt_is_causal=True) 
        
        logits = self.fc_out(decoder_output) # B X L X vocab_size

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return logits
    
    # Note: Output strips SOS and EOS tokens in result
    # will generate for whole batch
    def forward_test(self, image, max_length=30):
        # number of patches (from paper N = HW/P^2)
        N = (image.shape[2] * image.shape[3]) // (self.P**2)
    
        # batch num
        B = image.shape[0]

        embedded = self.make_patches(image) # B X N X (P * P * C)
        embedded = self.image_embedding_layer(embedded) * math.sqrt(self.embed_dim) # B X N X self.embed_dim
        embedded = self.positional_encoding(embedded)

        # start with just <SOS>
        generated = torch.full(
            (B, 1), # make a tensor for dimensions batch size, 1
            self.vocab.word_to_index[self.vocab.SOS_TOKEN], # make every row start with the Start of Sentence (SOS) token
            dtype=torch.long, device=image.device
        )
        
        encoder_output = self.transformer_encoder(embedded)
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