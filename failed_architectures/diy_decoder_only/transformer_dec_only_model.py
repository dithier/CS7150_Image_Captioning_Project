import math
from transformer_enc_doc_model import PositionalEncoding, MultiHeadAttention, FeedForwardNetwork
import torch
import torch.nn as nn

# Note when working with text BXLXC means B batches, L is sequence number, C is channels
# For images we have BXNXC where this time N is num patches
# We use these interchangeably in code below

class VisionTransformerDecoderCell(nn.Module):
    """
    A single cell (unit) for the Transformer decoder.
    """
    def __init__(self, input_dim: int, num_heads: int, ff_dim: int, dropout: float):
        """
        Inputs:
        - input_dim: Input dimension for each token in a sequence
        - num_heads: Number of attention heads in a multi-head attention module
        - ff_dim: The hidden dimension for a feedforward network
        - dropout: Dropout ratio for the output of the multi-head attention and feedforward
          modules.
        """
        super(VisionTransformerDecoderCell, self).__init__()

        ###########################################################################
        # This is similar to to version above but here we are doing PRE-NORM
        # A single Transformer encoder cell consists of
        # 1. Layer norm
        # 2. A multi-head attention module
        # 3. Followed by dropout
        # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm

        # At the same time, it also has
        # 1. Layer norm
        # 2. Followed by a feedforward network
        # 3. Followed by dropout
       
        ###########################################################################
        self.self_attn = MultiHeadAttention(input_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(input_dim) # I want it to be across the embedding dimension
        self.layer_norm_2 = nn.LayerNorm(input_dim) 

        self.feedforward = FeedForwardNetwork(input_dim, ff_dim) 

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None):
        """
        Inputs:
        - x: Tensor of the shape BxLxC, where B is the batch size, L is the sequence length,
            and C is the channel dimension.  
        - mask: Tensor for masking in the multi-head attention
        """

        ###########################################################################
        #  Get the output of the multi-head attention part (with dropout     #
        # and pre layer norm), which is used as input to layernorm and then the   #
        # feedforward network (again, followed by dropout).                       #
        #                                                                         #
        # Don't forget the residual connections for both parts.                   #
        ###########################################################################
        
        normalized = self.layer_norm_1(x)
        query, key, value = normalized, normalized, normalized 
        out = self.self_attn(query, key, value, mask) # mask should not be None
        out = self.dropout(out)
        c = x + out

        out = self.layer_norm_2(c)
        out = self.feedforward(out) 
        out = self.dropout(out)
        y = c + out
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return y


class VisionTransformerDecoder(nn.Module):
    """
    A full encoder consisting of a set of TransformerEncoderCell.
    """
    def __init__(self, input_dim: int, num_heads: int, ff_dim: int, num_cells: int, dropout: float=0.1):
        """
        Inputs:
        - input_dim: Input dimension for each token in a sequence
        - num_heads: Number of attention heads in a multi-head attention module
        - ff_dim: The hidden dimension for a feedforward network
        - num_cells: Number of TransformerEncoderCells
        - dropout: Dropout ratio for the output of the multi-head attention and feedforward
          modules.
        """
        super(VisionTransformerDecoder, self).__init__()

        ###########################################################################
        #  Construct a nn.ModuleList to store a stack of                     #
        # TranformerEncoderCells. Check the documentation here of how to use it   #
        # https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html#torch.nn.ModuleList

        # At the same time, define a layer normalization layer to process the     #
        # output of the entire encoder.                                           #
        ###########################################################################
        
        self.cells = nn.ModuleList([VisionTransformerDecoderCell(input_dim, num_heads, ff_dim, dropout) \
                                        for i in range(num_cells)])
        self.layer_norm = nn.LayerNorm(input_dim)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None):
        """
        Inputs:
        - x: Tensor of the shape BxLxC, where B is the batch size, L is the sequence length,
          and C is the channel dimension
        - encoder_output: The outputs of the encoder cell 
        - mask: Tensor for masking in the multi-head attention

        Return:
        - y: Tensor of the shape of BxLxC, which is the normalized output of the encoder
        """
        ###########################################################################
        # Feed x into the stack of TransformerEncoderCells and then         #
        # normalize the output with layer norm.                                   #
        ###########################################################################
        for layer in self.cells:
            x = layer(x, mask)

        y = self.layer_norm(x)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return y
    
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
        self.transformer_decoder = VisionTransformerDecoder(self.embed_dim, num_heads, 
                                                            trx_ff_dim, num_decoder_cells)
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
        # final mask size (N + L) X (N + L)
        dim = N + L
        mask = torch.zeros((dim, dim), dtype=torch.bool, device=device)

        # the same setup as the second slide for masking in lecture 14
        # makes the decoder not able to see future tokens during training (seq_len, seq_len)
        # our mutlihead attn block fills False/0 with -inf so we want the top right hand triangle
        # of our mask to be 0

        # we want the image part of the mask to be visible everywhere
        mask[:, :N] = 1

        # in this particular case of dim L X L we are applying the masking to bottom right hand corner that's
        # L X L
        causal_mask = (1 - torch.triu(torch.ones(L, L), diagonal=1)).bool()
        causal_mask = causal_mask.to(device)
        mask[N:, N:] = causal_mask

        # we also want to mask out padding tokens
        pad_mask = (labels != self.pad_token) # B X L

        # img tokens don't have pad token so we always want true
        pad_img_mask = torch.ones((labels.size(0), N), dtype=torch.bool, device=device) # B X N

        # concatenate pad masks
        pad_mask_final = torch.concat((pad_img_mask, pad_mask), dim=-1) # B X (N + L)

        # combine pad mask with mask
        #  (N + L) X (N + L) -> (1 , N + L, N + L).      B X (N + L) -> (B, 1, N + L)
        # we use & because we want BOTH entries in either mask to be true, otherwise it should be masked (False)
        mask = mask.unsqueeze(0) & pad_mask_final.unsqueeze(1)

        return mask.unsqueeze(1) # (B, 1, N + L, N + L)

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
        mask = self.get_mask(N, L, labels, image.device)

        # pass PE result into transformer decoder model and all its cells
        output = self.transformer_decoder(output, mask) 

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
    
   