import math

import torch
import torch.nn as nn

# Note when working with text BXLXC means B batches, L is sequence number, C is channels
# For images we have BXNXC where this time N is num patches
# We use these interchangeably in code below
class MultiHeadAttention(nn.Module):
    """
    A module that computes multi-head attention given query, key, and value tensors.
    """
    def __init__(self, input_dim: int, num_heads: int):
        """
        Constructor.

        Inputs:
        - input_dim: Dimension of the input query, key, and value. Here we assume they all have
          the same dimensions. But they could have different dimensions in other problems.
        - num_heads: Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()

        assert input_dim % num_heads == 0

        self.input_dim = input_dim
        self.num_heads = num_heads
        # channel dimension per attention head
        self.dim_per_head = input_dim // num_heads

        # for multiheadattn, here we assume input dim has already been cast to higher dimension
        # hence calculation for self.dim_per_head

        # note: I acknowledge self.dim_per_head * self.num_heads is just input dim, but writing
        # it out this way is more intuitive for how I think
        
        self.WQ = nn.Linear(input_dim, self.dim_per_head * self.num_heads) 
        self.WK = nn.Linear(input_dim, self.dim_per_head * self.num_heads)
        self.WV = nn.Linear(input_dim, self.dim_per_head * self.num_heads)

        self.output_layer = nn.Linear(self.dim_per_head * self.num_heads, \
                                      self.dim_per_head * self.num_heads)

        self.softmax = nn.Softmax(dim=3)


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor=None):
        """
        Compute the attended feature representations.

        Inputs:
        - query: Tensor of the shape BxLxC, where B is the batch size, L is the sequence length,
          and C is the channel dimension
        - key: Tensor of the shape BxLxC
        - value: Tensor of the shape BxLxC
        - mask: Tensor indicating where the attention should *not* be performed
        """
        b = query.shape[0]

        dot_prod_scores = None
        ###########################################################################
        # Compute the scores based on dot product between transformed query,#
        # key, and value. You may find torch.matmul helpful, whose documentation  #
        # can be found at                                                         #
        # https://pytorch.org/docs/stable/generated/torch.matmul.html#torch.matmul#
        # Remember to divide the dot product similarity scores by square root of  #
        # the channel dimension per head.
        #                                                                         #
        # Since no for loops are allowed here, think of how to use tensor reshape #
        # to process multiple attention heads at the same time.                   #
        ###########################################################################
        
        # Transform query, key, and value
        Q = self.WQ(query) # B x L x (D X H)
        K = self.WK(key)
        V = self.WV(value)

        # multihead split
        seq_len = Q.shape[1]
        
        Q = Q.view(b, -1, self.num_heads, self.dim_per_head) # shape B X L X H X D
        Q = torch.transpose(Q, 1, 2) # shape B X H X L X D

        K = K.view(b, -1, self.num_heads, self.dim_per_head) # shape B X L X H X D
        K = torch.transpose(K, 1, 2) # shape B X H X L X D

        V = V.view(b, -1, self.num_heads, self.dim_per_head) # shape B X L X H X D
        V = torch.transpose(V, 1, 2) # shape B X H X L X D

        # can't use K.t() because greater than 2 dimensional vector so need to explicitly call transpose
        # with indices where we swap last two dimensions
        K_transpose = torch.transpose(K, -1, -2) # B X H X D X L

        dot_prod_scores = (torch.matmul(Q, K_transpose)/ math.sqrt(self.dim_per_head)) # B X H X L X L
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        if mask is not None:
            # We simply set the similarity scores to be near negative infinity for
            # the positions where the attention should not be done. Think of why  #
            # we do this.
            dot_prod_scores = dot_prod_scores.masked_fill(mask == 0, -1e9)

        out = None
        ###########################################################################
        # Compute the attention scores, which are then used to modulate the #
        # value tensor. Finally concatenate the attended tensors from multiple    #
        # heads and feed it into the output layer. You may still find             #
        # torch.matmul helpful.                                                   #
        #                                                                         #
        # Again, think of how to use reshaping tensor to do the concatenation.    #
        ###########################################################################
        softmax = self.softmax(dot_prod_scores) # take it across columns in each batch
        Z = torch.matmul(softmax, V) # B X H X L X D

        Z = torch.transpose(Z, 1, 2) # B x L X H X D
        Z = torch.reshape(Z, (b, seq_len, -1))   # B X L X (H*D)
        
        out = self.output_layer(Z) 
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return out
       
class FeedForwardNetwork(nn.Module):
    """
    A simple feedforward network. Essentially, it is a two-layer fully-connected
    neural network.
    """
    def __init__(self, input_dim, ff_dim):
        """
        Inputs:
        - input_dim: Input dimension
        - ff_dim: Hidden dimension
        """
        super(FeedForwardNetwork, self).__init__()

        # https://piazza.com/class/mk46rb2r2tm2mj/post/71
        # above post says ignore dropout here

        ###########################################################################
        # Define the two linear layers and a non-linear one.
        ###########################################################################
        self.fc1 = nn.Linear(input_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, input_dim) 
        self.nonlinear = nn.ReLU() 
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def forward(self, x: torch.Tensor):
        """
        Input:
        - x: Tensor of the shape BxLxC, where B is the batch size, L is the sequence length,
         and C is the channel dimension

        Return:
        - y: Tensor of the shape BxLxC
        """

        y = None
        ###########################################################################
        # Process the input.                                                #
        ###########################################################################
        x = self.fc1(x)
        x = self.nonlinear(x)
        y = self.fc2(x)
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return y
    
class PositionalEncoding(nn.Module):
    """
    A module that adds positional encoding to each of the token's features.
    So that the Transformer is position aware.
    """
    def __init__(self, input_dim: int, max_len: int=10000):
        """
        Inputs:
        - input_dim: Input dimension about the features for each token
        - max_len: The maximum sequence length
        """
        super(PositionalEncoding, self).__init__()

        self.input_dim = input_dim

    def forward(self, x, max_length=10000):
        """
        Compute the positional encoding and add it to x.

        Input:
        - x: Tensor of the shape BxLxC, where B is the batch size, L is the sequence length,
          and C is the channel dimension (if image, L is N for num patches)
        - max_length: maximum sequence length the positional encoding can handle

        Return:
        - x: Tensor of the shape BxL(or N)xC, with the positional encoding added to the input
        """
        seq_len = x.shape[1]

        pe = None
        ###########################################################################
        # Compute the positional encoding                                   #
        # Check Section 3.5 for the definition (https://arxiv.org/pdf/1706.03762.pdf)
        #                                                                         #
        # It's a bit messy, but the definition is provided for your here for your #
        # convenience (in LaTex).                                                 #
        # PE_{(pos,2i)} = sin(pos / 10000^{2i/\dmodel}) \\                        #
        # PE_{(pos,2i+1)} = cos(pos / 10000^{2i/\dmodel})                         #
        #                                                                         #
        # You should replace 10000 with max_len here.
        ###########################################################################
        pe = torch.zeros((seq_len, self.input_dim))
        
        p = torch.tensor(range(seq_len)).unsqueeze(1)

        # even index
        pe[:, 0::2] = torch.sin(p / max_length**(torch.arange(0, self.input_dim, 2)/ self.input_dim)) 

        # odd index
        pe[:, 1::2] = torch.cos(p / max_length**(torch.arange(0, self.input_dim, 2)/ self.input_dim))

        ###########################################################################

        x = x + pe.to(x.device)
        return x
    
class VisionTransformerEncoderCell(nn.Module):
    """
    A single cell (unit) for the Transformer encoder.
    """
    def __init__(self, input_dim: int, num_heads: int, ff_dim: int, dropout: float):
        """
        Inputs:
        - input_dim: Input dimension for each token in a sequence/dim of image features
        - num_heads: Number of attention heads in a multi-head attention module
        - ff_dim: The hidden dimension for a feedforward network
        - dropout: Dropout ratio for the output of the multi-head attention and feedforward
          modules.
        """
        super(VisionTransformerEncoderCell, self).__init__()

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
        self.multihead_attn = MultiHeadAttention(input_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(input_dim) # I want it to be across the embedding dimension
        self.layer_norm_2 = nn.LayerNorm(input_dim)

        self.feedforward = FeedForwardNetwork(input_dim, ff_dim) 
        

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None):
        """
        Inputs:
        - x: Tensor of the shape BxNxC, where B is the batch size, N is number of patches,
            and C is the channel dimension
        - mask: Tensor for masking in the multi-head attention
        """

        y = None
        ###########################################################################
        #  Get the output of the multi-head attention part (with dropout     #
        # and pre layer norm), which is used as input to layernorm and then the   #
        # feedforward network (again, followed by dropout).                       #
        #                                                                         #
        # Don't forget the residual connections for both parts.                   #
        ###########################################################################
        normalized = self.layer_norm_1(x)
        query, key, value = normalized, normalized, normalized 
        out = self.multihead_attn(query, key, value, mask)
        out = self.dropout(out)
        b = x + out

        out = self.layer_norm_2(b)
        out = self.feedforward(out) 
        out = self.dropout(out)
        y = b + out

        return y
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
        self.cross_attn = MultiHeadAttention(input_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(input_dim) # I want it to be across the embedding dimension
        self.layer_norm_2 = nn.LayerNorm(input_dim) 
        self.layer_norm_3 = nn.LayerNorm(input_dim) 

        self.feedforward = FeedForwardNetwork(input_dim, ff_dim) 

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, mask: torch.Tensor=None):
        """
        Inputs:
        - x: Tensor of the shape BxLxC, where B is the batch size, L is the sequence length,
            and C is the channel dimension. 
        - encoder_output: Tensor of the shape BxLxC, where B is the batch size, L is the sequence length,
            and C is the channel dimension. This is coming from the ENCODER
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

        normalized = self.layer_norm_2(c)
        query = normalized
        key, value = encoder_output, encoder_output
        out = self.cross_attn(query, key, value, None)
        out = self.dropout(out)
        b = c + out

        out = self.layer_norm_3(b)
        out = self.feedforward(out) 
        out = self.dropout(out)
        y = b + out
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return y

class VisionTransformerEncoder(nn.Module):
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
        super(VisionTransformerEncoder, self).__init__()

        ###########################################################################
        #  Construct a nn.ModuleList to store a stack of                     #
        # TranformerEncoderCells. Check the documentation here of how to use it   #
        # https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html#torch.nn.ModuleList

        # At the same time, define a layer normalization layer to process the     #
        # output of the entire encoder.                                           #
        ###########################################################################
        
        self.cells = nn.ModuleList([VisionTransformerEncoderCell(input_dim, num_heads, ff_dim, dropout) \
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
        - mask: Tensor for masking in the multi-head attention

        Return:
        - y: Tensor of the shape of BxLxC, which is the normalized output of the encoder
        """

        y = None
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

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, mask: torch.Tensor=None):
        """
        Inputs:
        - x: Tensor of the shape BxLxC, where B is the batch size, L is the sequence length,
          and C is the channel dimension
        - encoder_output: The outputs of the encoder cell 
        - mask: Tensor for masking in the multi-head attention

        Return:
        - y: Tensor of the shape of BxLxC, which is the normalized output of the encoder
        """

        y = None
        ###########################################################################
        # Feed x into the stack of TransformerEncoderCells and then         #
        # normalize the output with layer norm.                                   #
        ###########################################################################
        for layer in self.cells:
            x = layer(x, encoder_output, mask)

        y = self.layer_norm(x)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return y
    
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
        self.transformer_encoder = VisionTransformerEncoder(self.embed_dim, num_heads, trx_ff_dim,
                                                      num_encoder_cells, dropout)
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

        embedded = self.make_patches(image) # B X N X (P * P * C)
        embedded = self.image_embedding_layer(embedded) * math.sqrt(self.embed_dim) # B X N X self.embed_dim

        # the same setup as the second slide for masking in lecture 14
        # makes the decoder not able to see future tokens during training (seq_len, seq_len)
        # our mutlihead attn block fills False/0 with -inf so we want the top right hand triangle
        # of our mask to be 0
        seq_len = labels.size(1)

        # handles padding tokens we don't want our decoder to see (B, 1, 1, seq_len)
        # again bc multihead attn block fills False/0 with -inf we want where our pad
        # tokens are to be false
        pad_mask = (labels != self.pad_token).unsqueeze(1).unsqueeze(2)

        causal_mask = (1 - torch.triu(torch.ones(seq_len, seq_len), diagonal=1)).bool().to(labels.device)

        mask = pad_mask | causal_mask 

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
        label_embeddings = self.positional_encoding(label_embeddings)
        decoder_output = self.transformer_decoder(label_embeddings, encoder_output, mask) 
        logits = self.fc_out(decoder_output) # B X L X vocab_size

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return logits
    
    # Note: Output strips SOS and EOS tokens in result
    # this is in english
    # will generate for whole batch
    # TODO: THIS IS OLD IMPLEMENTATION-- NEED TO FIX
    def forward_test(self, image, max_length=30):
        # number of patches (from paper N = HW/P^2)
        N = (image.shape[2] * image.shape[3]) // (self.P**2)
    
        # batch num
        B = image.shape[0]

        embedded = self.make_patches(image) # B X N X (P * P * C)
        embedded = self.image_embedding_layer(embedded) * math.sqrt(self.embed_dim) # B X N X self.embed_dim
        embedded = self.positional_encoding(embedded)

        B = image.size(0)

        # start with just <SOS>
        generated = torch.full(
            (B, 1), # make a tensor for dimensions batch size, 1
            self.vocab.word_to_index[self.vocab.SOS_TOKEN], # make every row start with the Start of Sentence (SOS) token
            dtype=torch.long, device=image.device
        )
        
        encoder_output = self.transformer_encoder(embedded)
        # print(f"encoder output mean: {encoder_output.mean().item():.4f}")
        # print(f"encoder output std:  {encoder_output.std().item():.4f}")
        decoder_outputs = []

        for _ in range(max_length + 1):
            # todo: No embedding and pe needed?
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
    

        