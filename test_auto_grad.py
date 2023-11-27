import torch

a = torch.tensor([2., 3.])
a.requires_grad = True
b = torch.tensor([4., 5.])
b.requires_grad = True

Q = 3 * a ** 3 - b ** 2