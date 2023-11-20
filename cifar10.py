import torch
import torchvision
import torchvision.transforms as transforms


tranform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data',train=True, download=True,transform=tranform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data',train=True, download=True,transform=tranform)
testloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

