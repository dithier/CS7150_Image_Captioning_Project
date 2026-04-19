import math
from diy_transformer_enc_dec.transformer_enc_dec_model import PositionalEncoding
import torch
import torch.nn as nn
class VisionTransformerDecoderModel(nn.Module):
    """
    A Transformer-based image captioning model
    """
    def __init__(self,
            vocab, P: int, embed_dim: int, num_heads: int, trx_ff_dim: int,
            num_decoder_cells : int, dropout: float=0.1
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
        super(VisionTransformerDecoderModel, self).__init__()

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

        # Note: becauses no cross attention we use pytorch's encoder cell for our decoder
        layer_norm_1 = nn.LayerNorm(self.embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(self.embed_dim, num_heads, dim_feedforward=trx_ff_dim,
                                                   dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_decoder = nn.TransformerEncoder(encoder_layer, num_layers=num_decoder_cells, norm=layer_norm_1,
                                                         enable_nested_tensor=False) 
        
        self.embedding = nn.Embedding(len(vocab), self.embed_dim)
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

    def get_mask(self, N, L, labels, device):
        # final causal mask size (N + L) X (N + L)
        dim = N + L
        causal_mask = torch.zeros((dim, dim), dtype=torch.bool, device=device)

        # True in pytorch attn mask means you are not allowed to attend (reverse from HWs)

        # we want the image part of the mask to be visible everywhere
        # so causal_mask[:, :N] remains the initialized 0

        # However, the picture tokens shouldn't be allowed to attend to labels
        causal_mask[:N, N:] = 1

        submask = (torch.triu(torch.ones(L, L), diagonal=1)).bool()
        submask = submask.to(device)
        causal_mask[N:, N:] = submask

        # we also want to mask out padding tokens
        pad_mask = (labels == self.pad_token) # B X L
        # no padding during image part of input so all should be seen by model
        pad_img_mask = torch.zeros((labels.size(0), N), dtype=torch.bool, device=device)
        # concatenate pad masks
        pad_mask = torch.concat((pad_img_mask, pad_mask), dim=-1) # B X (N + L)

        return pad_mask, causal_mask

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

        # Num in seq
        L = labels.size(1)

        image_tokens = self.make_patches(image) # B X N X (P * P * C)
        
        # C' is embed dimension
        image_embedding = self.image_embedding_layer(image_tokens)  * math.sqrt(self.embed_dim) # B X N X C'

        # label embeddings B X L
        label_embeddings = self.embedding(labels) * math.sqrt(self.embed_dim) # B X L X C'
        
        # Now we concatenate our image and label embeddings by making the image embedding a prefix
        embedding = torch.cat((image_embedding, label_embeddings), dim=1) # B x (N + L) X C'
    
        # positional encoding applied to whole combined embedding
        output = self.positional_encoding(embedding) # B X (N + L) X C'

        # generate mask for next step
        pad_mask, causal_mask = self.get_mask(N, L, labels, image.device)

        # pass PE result into transformer decoder model and all its cells
        output = self.transformer_decoder(output, mask=causal_mask, src_key_padding_mask=pad_mask,
                                          is_causal=True) 
    

        # get just positions related to captions in output
        caption_output = output[:, N:, :]

        logits = self.fc_out(caption_output) # B X L X vocab_size

        return logits
    
    # Note: Output strips SOS and EOS tokens in result
    # will generate for whole batch
    def forward_test(self, image, max_length=30):
        # number of patches (from paper N = HW/P^2)
        N = (image.shape[2] * image.shape[3]) // (self.P**2)
    
        # batch num
        B = image.shape[0]

        embedded = self.make_patches(image) # B X N X (P * P * C)
        embedded = self.image_embedding_layer(embedded) # B X N X self.embed_dim

        # start with just <SOS>
        start = torch.full(
            (B, 1), # make a tensor for dimensions batch size, 1
            self.vocab.word_to_index[self.vocab.SOS_TOKEN], # make every row start with the Start of Sentence (SOS) token
            dtype=torch.long, device=image.device
        )
        
        # generated is embedded image and embedded start token, both scaled 
        generated = torch.cat((embedded, self.embedding(start)), dim=1) * math.sqrt(self.embed_dim)

        decoder_outputs = []

        for _ in range(max_length + 1):
            # this includes image embedding and tokens generated so far
            decoder_input = self.positional_encoding(generated)
            decoder_output = self.transformer_decoder(decoder_input)

            output = self.fc_out(decoder_output)
            last_output = output[:, -1, :]

            decoder_outputs.append(last_output)

            # get indices of highest values 
            _, topi = last_output.topk(1)

            top_i_embedded = self.embedding(topi) * math.sqrt(self.embed_dim)
        
            generated = torch.cat((generated, top_i_embedded), dim=1)
        
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
    
   