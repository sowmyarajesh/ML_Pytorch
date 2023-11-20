'''
Export a pytorch model to onnx model 
This code requires the transfer_learning.py executed for the model to be saved  in the location mentioned here. 
Otherwise, model path and data path need to be updated in the code. 
Dataset: https://www.kaggle.com/datasets/ajayrana/hymenoptera-data
ref: https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

'''
import torch
import numpy as np
import os
import time
from torchvision import datasets, models, transforms

# setting device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Hyper parameters
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])
batch_size = 1

#load data for validation
data_dir = 'data/hymenoptera_data'
data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
testData = datasets.ImageFolder(os.path.join(data_dir, 'val'),data_transform)
dataloader = torch.utils.data.DataLoader(testData, batch_size=4, shuffle=True, num_workers=0)
x, classes = next(iter(dataloader)) # get one batch

# load model
start = time.time()
model = torch.load('model/transfer_learning.pth')
end = time.time()
print(f"time to load pytorch model = {end-start}")
torch_model = model.eval() # load the model in evaluation mode
torch_out = torch_model(x)
# export as onnx model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "model/transfer_learning.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                    'output' : {0 : 'batch_size'}})

## Load the Onnx model 
import onnx
start =time.time()
onnx_model = onnx.load("model/transfer_learning.onnx")
end = time.time()
print(onnx.checker.check_model(onnx_model))
print(f"time to load onnc model = {end-start}")