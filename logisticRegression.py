'''
Pytorch pipeline
----------------
0) Prepare data
1) Design model (input size, output size, forward pass)
2) loss and optimizer
3) Training loop
    # forward pass
    # backward pass
    # update weights
'''
import torch
from torch import nn 
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) Prepare data
bc = datasets.load_breast_cancer()
X,y = bc.data, bc.target
n_samples, n_features = X.shape
print(n_samples,n_features)

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=1234)
### Scale the feature to have 0 mean 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

### Convert to torch tensor
X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0],1) # convert 1d to 2d
y_test = y_test.view(y_test.shape[0],1) # convert 1d to 2d


# 1) Design model (input size, output size, forward pass)
class LogisticRegression(nn.Module):
    def __init__(self, n_input_feats):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_feats,1)
    
    def forward(self,x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
model = LogisticRegression(n_features)

# 2) loss and optimizer
criterion = nn.BCELoss() ## Binary Cross entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 3) Training loop
n_iter = 100
for epoch in range(n_iter):
    # forward pass
    y_pred = model(X_train)
    loss = criterion(y_pred,y_train)

    # backward pass
    loss.backward()

    # update weights
    optimizer.step()
    optimizer.zero_grad()
    if (epoch+1%10==0):
        print(f'epoch: {epoch+1}, loss= {loss.item():4f}')

with torch.no_grad():
    y_pred = model(X_test)
    y_pred_cls = y_pred.round()
    acc = y_pred_cls.eq(y_test).sum()/float(y_pred.shape[0])
    print(f'accuracy = {acc:.4f}')