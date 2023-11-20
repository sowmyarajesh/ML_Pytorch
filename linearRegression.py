import torch
from torch import nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) Prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100,n_features=1,noise=20, random_state=1)
### default will be double. we need to change it to float32 to avoid memory issues.
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
### Reshape the data. convert the 1 X n to n X 1 array
y = y.view(y.shape[0],1)
n_samples, n_features = X.shape

# 1) Design model (input size, output size, forward pass)
input_size = n_features
output_size =1
model = nn.Linear(input_size, output_size)

# 2) loss and optimizer
lr = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


# 3) Training loop
n_iter = 100
for epoch in range(n_iter):
    # forward pass
    y_pred = model(X)
    loss = criterion(y_pred,y)

    # backward pass
    loss.backward()

    # update weights
    optimizer.step()
    optimizer.zero_grad()
    if epoch+1%10==0:
        print(f'epoch: {epoch+1}, loss = {loss.item()}:.4f')

# plot
predicted = model(X).detach()
plt.plot(X_numpy,y_numpy, 'ro')
plt.plot(X_numpy,predicted,'bo')
plt.show()



