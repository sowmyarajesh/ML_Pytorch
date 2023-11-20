import torch
import numpy as np


tensor = torch.tensor([1, 2, 3])

# # Perform basic operations on it
# print("Add Example")
# print(tensor + 10)
# print("Multiply Example")
# print(tensor * 10)
# print("Subtract Example")
# print(tensor - 10)
# print("Divide Example")
# print(tensor / 10)


# # perform basic operations using tensor functions. 
# print(torch.mul(tensor,10))
# print(torch.add(tensor,10))
# print(torch.sub(tensor,3))
# print(torch.div(tensor,10))
# print(torch.multiply(tensor, tensor)) # element wise multiplication
# print(torch.mul(tensor, tensor)) # element wise multiplication
# print(torch.matmul(tensor, tensor)) # matrix multiplication
# print(tensor @ tensor) # matrix multiplication

# Shapes need to be in the right way  
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)

# print(torch.matmul(tensor_A, tensor_B)) # (this will error)

'''
The main two rules for matrix multiplication to remember are:

    The inner dimensions must match:
        (3, 2) @ (3, 2) won't work
        (2, 3) @ (3, 2) will work
        (3, 2) @ (2, 3) will work
    The resulting matrix has the shape of the outer dimensions:
        (2, 3) @ (3, 2) -> (2, 2)
        (3, 2) @ (2, 3) -> (3, 3)

'''
# print(torch.matmul(tensor_A,tensor_B.T))
# # torch.mm is a shortcut for matmul
# print(torch.mm(tensor_A, tensor_B.T))

'''
    The torch.nn.Linear() module also known as a feed-forward layer or fully connected layer, 
    implements a matrix multiplication between an input x and a weights matrix A.
    y = x.(A.T)+b
'''

# # Since the linear layer starts with a random weights matrix, let's make it reproducible (more on this later)
# torch.manual_seed(42)
# # This uses matrix multiplication
# linear = torch.nn.Linear(in_features=2, # in_features = matches inner dimension of input 
#                          out_features=6) # out_features = describes outer value 
# x = tensor_A
# output = linear(x)
# print(f"Input shape: {x.shape}\n")
# print(f"Output:\n{output}\n\nOutput shape: {output.shape}")

'''
performing aggregation functions
'''
# # Create a tensor
# x = torch.arange(0, 100, 10)
# print(f"Minimum: {x.min()}")
# print(f"Maximum: {x.max()}")
# # print(f"Mean: {x.mean()}") # this will error
# print(f"Mean: {x.type(torch.float32).mean()}") # won't work without float datatype
# print(f"Sum: {x.sum()}")

# # find the index of a tensor where the max or minimum occurs
# print(f"Index where max value occurs: {x.argmax()}")
# print(f"Index where min value occurs: {x.argmin()}")

'''
Slicing and reshaping existing tensor array
'''

# # slicing operations
# x = torch.rand(5,3)
# print(x[:,0])
# print(x[1:])


# # reshaping tensor
# x = torch.rand(4,4)
# print(x)
# y=x.view(16) # 4*4 is 16 we will be represent the result as 1 dim
# print(y)
# y  = x.view(-1,8) # create a 2X8 array
# print(y)


'''
tensor - numpy 
We can create tensor from numpy array. Numpy array can be created from tensors. 
If the tensor is in CPU, both tensor and numpy object will share the same location in the memory.
When we change one, the other will also be changed. 


'''
# x = np.ones(5)
# print(x)
# y = torch.from_numpy(x)
# print(y)

# x+=1
# print(x)
# print(y)
# we could see that update in x will affect y too. This will happen only in CPU.

# if torch.cuda.is_available(): # checking out GPU tensor
#     device = torch.device("cuda")
#     x = torch.ones(5, device=device)
#     y = torch.ones(5)
#     y = y.to(device)
#     z = x+y
#     z = z.to("cpu") # numpy can only handle cpu tensor. it will throw error when processed GPU tensor.
#     z = z.numpy() # converts tensor to numpy
#     y1 = x
#     y1 = y1.to("cpu")
#     x+=1
#     print(x)
#     print(y1)


'''
Sometimes, we will apply gradiations to tensor variables during model optimization. 
For those variables, we can mention that during definition
'''
z = torch.ones(5, requires_grad=True)  # default is False


