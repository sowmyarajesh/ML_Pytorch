import torch
# print(torch.__version__)


# introduction to tensors
'''
 Tensors are ways to represent the multi dimensional data
'''
# # create tensor
# # scalar
# scalar = torch.tensor(7)
# print(scalar)
# print(scalar.item())
# print(scalar.ndim)

# # vector
# vector = torch.tensor([3,3])
# print(vector)
# print(vector.shape)
# print(vector.ndim)

# # matrix
# MATRIX = torch.tensor([[1,2],[3,4]])
# print(MATRIX)
# print(MATRIX.ndim)
# print(MATRIX.shape)


# Random 
rnd_tensor = torch.rand((2,2))
print(rnd_tensor)
print(rnd_tensor.ndim)


# # Image as tensor
# # Create a random tensor of size (224, 224, 3) => [height, width, color_channels(r,g,b)]
# random_image_size_tensor = torch.rand(size=(224, 224, 3))
# random_image_size_tensor.shape, random_image_size_tensor.ndim


# # Create a tensor of all zeros
# zeros = torch.zeros(size=(3, 4))
# print(zeros)
# print(zeros.dtype)


# # Create a tensor of all ones
# ones = torch.ones(size=(3, 4))
# print(ones) 
# print(ones.dtype)


# Create a range of values 0 to 10
zero_to_ten = torch.arange(start=0, end=10, step=1)
print(zero_to_ten)


# Can also create a tensor of zeros similar to another tensor
ten_zeros = torch.zeros_like(input=zero_to_ten) # will have same shape
print(ten_zeros)

# getting information from tensor
'''
Generally if you see torch.cuda anywhere, 
the tensor is being used for GPU 
(since Nvidia GPUs use a computing toolkit called CUDA).
'''
# Create a tensor with random values
some_tensor = torch.rand(3, 4)

# Find out details about it
print(some_tensor)
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}") # will default to CPU