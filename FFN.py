'''
Feed Forward network
-----------------------
Dataset : MNIST
Steps:
# Data loader, Transformation
# Multilayer Neural network, activation function
# Loss and optimizer
# Training loop (Batch training)
# Model evaluation
# GPU support

'''
import numpy as np
import torch
from torch import nn 
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784 # 28x28
hidden_size = 500 
n_class = 10
n_iter = 2
batch_size = 100
learning_rate = 0.001

# MNIST dataset , We will store the data into data folder locally
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader, Transformation
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

## Get one batch from this data set
example = iter(train_loader)
samples,labels = next(example)
print(samples.shape, labels.shape)
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0], cmap='gray') # samples[i][0] -> we want to access only the first channel
plt.show()

# Multilayer Neural network, activation function
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size,num_class)
    
    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out
    
model = NeuralNet(input_size, hidden_size, n_class)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Training loop (Batch training)
n_steps = len(train_loader)
for epoch in range(n_iter):
    for i , (images, labels)  in enumerate(train_loader):
        ## 100 X 1 X 28 X 28
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        ## forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        ## backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch+1%10==0:
            print(f'epoch: {epoch+1}, loss = {loss.item():.4f}') 
        
# Model evaluation
## test
with torch.no_grad():
    n_pos = 0
    n_samples =0
    for images, labels in test_loader:
        images = images.reshape(-1,28*28).to(device)
        labels= labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs,1)
        n_samples += labels.shape[0]
        n_pos += (preds==labels).sum().item()

accuracy = 100* n_pos/n_samples
print("accuracy = ", accuracy)

