'''
Here  will try show the gradient descent algorithm using pytorch 
'''
import numpy as np
import torch

''' step -1:  Manually perform the task'''

# # f = w *x
# # y = 2*x

# X = np.array([1,2,3,4],dtype=np.float32)
# Y = np.array([2,4,6,8],dtype=np.float32)
# w = 0.0

# # model prediction
# def forward(x):
#     return w*X
# # loss = MSE = (1/N)*((w*x -y)**2)
# def loss(y, y_pred):
#     return ((y_pred - y)**2).mean()

# # gradient = dJ/dw == (1/N)*((2*x).(w*x - y))

# def gradient(x,y,y_pred):
#     return np.dot(2*x, y_pred-y).mean()


# print('prediction before training = {}'.format(forward(5)))

# #training
# learning_rate = 0.01
# n_iter = 10
# for epoch in range(n_iter):
#     #pred = forward_pass

#     y_pred = forward(X)
#     l = loss(Y,y_pred)
#     dw = gradient(X,Y,y_pred) # gradeint = backward pass
#     w-=learning_rate*dw # update weight
#     if epoch%1==0:
#         print("epoch {}: w - {}, loss= {}".format(epoch+1,w,l))
        
# print('prediction after training = {}'.format(forward(5)))


'''
Step -2 Try the same with Torch
'''
# X = torch.tensor([1,2,3,4], dtype=torch.float32)
# Y = torch.tensor([2,4,6,8],dtype=torch.float32)
# w = torch.tensor(0.0, dtype=torch.float32, requires_grad= True)

# # model prediction
# def forward(x):
#     return w*X
# # loss = MSE = (1/N)*((w*x -y)**2)
# def loss(y, y_pred):
#     return ((y_pred - y)**2).mean()


# print('prediction before training = {}'.format(forward(5)))

# #training
# learning_rate = 0.01
# n_iter = 20
# for epoch in range(n_iter):
#     #pred = forward_pass
#     y_pred = forward(X)
#     l = loss(Y,y_pred)
#     l.backward() # this will calculate the gradient
#     with torch.no_grad():
#         w-=learning_rate*dw
#     # empty the gradients (reset) after each iteration
#     w.grad.zero_()
#     if epoch%1==0:
#         print("epoch {}: w - {}, loss= {}".format(epoch+1,w,l))
        
# print('prediction after training = {}'.format(forward(5)))



'''
Step -3 Change the manually computed loss to automated loss completetion
Strp -4 automate the optimizer
# '''


# X = torch.tensor([1,2,3,4], dtype=torch.float32)
# Y = torch.tensor([2,4,6,8],dtype=torch.float32)
# w = torch.tensor(0.0, dtype=torch.float32, requires_grad= True)

# # model prediction
# def forward(x):
#     return w*x

# print('prediction before training = {}'.format(forward(5)))

# # Training
# learning_rate = 0.01
# n_iter = 20
# loss = torch.nn.MSELoss()

# optimizer = torch.optim.SGD([w],lr=learning_rate)

# for epoch in range(n_iter):
    
#     y_pred = forward(X) # pred = forward_pass
#     l = loss(Y,y_pred)
#     l.backward() # this will calculate the gradient
#     optimizer.step() # update weights
#     w.grad.zero_() # reset weights
#     if epoch%1==0:
#         print("epoch {}: w - {}, loss= {}".format(epoch+1,w,l))
        
# print('prediction after training = {}'.format(forward(5)))


'''
Step - 5  Use the default Model. 
'''


# X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
# Y = torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)
# X_test = torch.tensor([[5]], dtype=torch.float32)
# Y_test = torch.tensor([[10]],dtype=torch.float32)


# n_samples , n_features = X.shape
# print(n_samples,n_samples)


# # Model design:
# input_size = n_features
# output_size = n_features
# model = torch.nn.Linear(input_size,output_size)


# print('prediction before training = {}'.format(model(X_test).item()))

# # Training
# learning_rate = 0.03
# n_iter = 15
# loss = torch.nn.MSELoss()

# optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

# for epoch in range(n_iter):
    
#     y_pred = model(X) # pred = forward_pass
#     l = loss(Y,y_pred)
#     l.backward() # this will calculate the gradient
#     optimizer.step() # update weights
#     if epoch%1==0:
#         [w,b] = model.parameters()
#         print("epoch {}: w - {}, loss= {}".format(epoch+1,w[0].item(),l))
        
# print('prediction after training = {}'.format(model(X_test).item()))


'''
Simplify by moving the model definition into a class
'''

class LinearRegression(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define layer
        self.lin = torch.nn.Linear(input_dim,output_dim)
    
    def forward(self, x):
        return self.lin(x)
    
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)
X_test = torch.tensor([[5]], dtype=torch.float32)
Y_test = torch.tensor([[10]],dtype=torch.float32)


n_samples , n_features = X.shape
print(n_samples,n_samples)


# Model design:
input_size = n_features
output_size = n_features
model = LinearRegression(input_size,output_size)


print('prediction before training = {}'.format(model.forward(X_test).item()))

# Training
learning_rate = 0.03
n_iter = 15
loss = torch.nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(n_iter):
    
    y_pred = model.forward(X) # pred = forward_pass
    l = loss(Y,y_pred)
    l.backward() # this will calculate the gradient
    optimizer.step() # update weights
    if epoch%1==0:
        [w,b] = model.parameters()
        print("epoch {}: w - {}, loss= {}".format(epoch+1,w[0].item(),l))
        
print('prediction after training = {}'.format(model.forward(X_test).item()))


