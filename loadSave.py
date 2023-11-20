import torch
model = torch.load('model/transfer_learning.pth')
print(model.state_dict())
