import torch
import numpy as np

'''
Gradients are essential for model optimization. Pytorch has auograd package for that.


In the background, it will create a  vector jacovian product: 
Jacovian matrix with partial derivative and multiply it with the gradient vector to get the final gradients.
This is called chain rule
 [[dy_1/dx_1,....dy_m/dx_1],...,[dy_1/dx_n,....dy_m/dx_n]].[dl/dy_1,...dl/dy_m] = (dl/dx_1,...dl/dx_n)
'''

# n=3
# x = torch.ones(n, requires_grad=True)
# print(x)

# y = x+2
# print(y)

# z = y*y*2
# print(z)

# z=z.mean()
# print(z)

# z.backward() # dz/dx
# print(x.grad)


'''
If we want to remove the gradiation in a variable
x.requires_grad(False)
x.detach()
The above two remove the gradiant function from the variable. 
If you want to use the variable without gradient function, but do not want to remove it completely,

'''
# with torch.no_grad():
#     y = x+2
#     print(y) # will not have gradient function.


'''
Example
'''

weights = torch.ones(4,requires_grad=True)
for epoch in range(2):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_() # empty gradients for next iteration
    