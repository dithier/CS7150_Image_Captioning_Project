import seaborn
import torch
import matplotlib.pyplot as plt

device = "cpu"
PAD_TOKEN = 0

def get_mask_dec(N, L, labels, device):
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
        pad_mask = (labels == PAD_TOKEN) # B X L
        # no padding during image part of input so all should be seen by model
        pad_img_mask = torch.zeros((labels.size(0), N), dtype=torch.bool, device=device)
        # concatenate pad masks
        pad_mask = torch.concat((pad_img_mask, pad_mask), dim=-1) # B X (N + L)

        return pad_mask, causal_mask

labels = torch.tensor([[1, 5, 20, 22, 39, 2, 0, 0, 0, 0],
                       [1, 30, 2, 0, 0, 0, 0, 0, 0, 0]])


N = 5 
B = labels.size(0)
L = labels.size(1)


# (B, 1, N + L, N + L)
pad_mask, causal_mask = get_mask_dec( N, L, labels, device)

for i in range(B):
    print(pad_mask[i])

plt.figure(figsize=(5,5))
plt.imshow(causal_mask.numpy())

print(f"0,0: {causal_mask[0,0]}")


causal_mask_N = torch.triu(torch.ones(labels.size(1), labels.size(1)), diagonal=1).bool().to(labels.device)

plt.figure(figsize=(5,5))
plt.imshow(causal_mask_N.numpy())