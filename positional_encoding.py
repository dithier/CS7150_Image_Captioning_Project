"""
Authors: Priyanshu Ranka,  Carter Ithier
Course: CS 7150 - Deep Learning
Semester: Spring 2026
Short description:  Positional encoding module for Transformer models. 
"""

import torch.nn as nn
import torch

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