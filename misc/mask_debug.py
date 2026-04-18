
import seaborn
import torch
import matplotlib.pyplot as plt

def create_causal_mask(size):
    "Mask out subsequent positions. Also known as a causual mask."
    return (1 - torch.triu(torch.ones(size, size), diagonal=1)).bool()
    
    
# Let's visualize what the target mask looks like
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")

PAD_TOKEN = 0
# B x L (1, 10)
labels = torch.tensor([[1, 5, 20, 22, 39, 2, 0, 0, 0, 0],
                       [1, 30, 2, 0, 0, 0, 0, 0, 0, 0]])

B = labels.size(0)
seq_len = labels.size(1)

causal_mask = create_causal_mask(seq_len)
print(causal_mask.shape)
print(causal_mask[0,0])

# plt.figure(figsize=(5,5))
# plt.imshow(causal_mask.numpy())

#  self.word_to_index = {
        #     self.PAD_TOKEN: 0,
        #     self.SOS_TOKEN: 1,
        #     self.EOS_TOKEN: 2,
        #     self.UNK_TOKEN: 3,
        # }

pad_mask = (labels != PAD_TOKEN).unsqueeze(1).unsqueeze(2)




mask = causal_mask & pad_mask

mask1 = mask[0, 0, :, :]
mask2 = mask[1, 0, :, :]

plt.figure(figsize=(5,5))
plt.imshow(mask1.numpy())

plt.figure(figsize=(5,5))
plt.imshow(mask2.numpy())