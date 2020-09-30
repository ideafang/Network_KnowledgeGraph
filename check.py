import torch
import torch.nn.functional as F

label = [3, 5, 1, 4]
label = torch.tensor(label)
label = F.one_hot(label)
print(label)
