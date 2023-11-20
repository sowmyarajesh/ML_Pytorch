'''
loads the saved pytorch model from the given location. 
For the file mentioned in this code, transfer_learning.py need to be executed to generate the model and save in the location. 
'''
import torch
model = torch.load('model/transfer_learning.pth')
print(model.state_dict())
