import seaborn
import torch
import matplotlib.pyplot as plt

device = "cpu"
PAD_TOKEN = 0

def get_mask(N, L, labels, device):
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

        # plt.figure(figsize=(5,5))
        # plt.imshow(mask.numpy())

        # we also want to mask out padding tokens
        pad_mask = (labels != PAD_TOKEN) # B X L

        # B X 1 X 1 X L
        # pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)

        # img tokens don't have pad token so we always want true
        pad_img_mask = torch.ones((labels.size(0), N), dtype=torch.bool, device=device) # B X N


        # concatenate pad masks
        pad_mask_final = torch.concat((pad_img_mask, pad_mask), dim=-1) # B X (N + L)

        # plt.figure(figsize=(5,5))
        # plt.imshow(pad_mask_final.numpy())

        # combine pad mask with mask
        #  (N + L) X (N + L) -> (1 , N + L, N + L).      B X (N + L) -> (B, 1, N + L)
        # we use & because we want BOTH entries in either mask to be true, otherwise it should be masked (False)
        mask = mask.unsqueeze(0) & pad_mask_final.unsqueeze(1)

        return mask.unsqueeze(1) # (B, 1, N + L, N + L)




# B x L (1, 10)
labels = torch.tensor([[1, 5, 20, 22, 39, 2, 0, 0, 0, 0],
                       [1, 30, 2, 0, 0, 0, 0, 0, 0, 0]])

# labels = torch.tensor([[1, 5, 20, 22, 39, 2, 0, 0, 0, 0]])

N = 5 
B = labels.size(0)
L = 10 #labels.size(1)


# (B, 1, N + L, N + L)
mask = get_mask( N, L, labels, device)

print(mask.shape)

mask1 = mask[0, 0, :, :]
mask2 = mask[1, 0, :, :]

plt.figure(figsize=(5,5))
plt.imshow(mask1.numpy())

plt.figure(figsize=(5,5))
plt.imshow(mask2.numpy())