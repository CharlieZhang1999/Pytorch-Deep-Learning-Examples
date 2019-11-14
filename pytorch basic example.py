import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

x = torch.randn(10,3)
y = torch.randn(10,2)
linear = nn.Linear(3,2)

print('b',linear.bias)
print('w',linear.weight)
print(linear.weight.shape)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(),lr=0.01)
pred = linear(x)
loss = criterion(pred,y)
print('loss: ', loss.item())
for i in range(1,10):
    pred = linear(x)

    loss = criterion(pred,y)
    ##print('loss: ', loss.item())
    loss.backward()
    print('dL/dw',linear.weight.grad)
    print('dL/db',linear.bias.grad)

    optimizer.step()

pred = linear(x)
loss = criterion(pred,y)
print('loss: ', loss.item())

#pytorch basics