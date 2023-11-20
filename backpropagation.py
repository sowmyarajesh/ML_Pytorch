import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

#forward pass  and compute the loss
y_ht = w*x
loss = (y_ht - y)**2
print(loss)

#backward pass
loss.backward()
print(w.grad)

### update weight
### next iteration of forward and backward