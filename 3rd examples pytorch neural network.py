"""
Created on Mon Nov 18 21:22:32 2019

@author: djogem
"""

import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
#import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\

input_size = 784
hidden_layer = 500
num_class = 10
num_epoch = 10
batch_size = 100#64
learning_rate = 0.001

train_dataset = torchvision.datasets.MNIST(root='../../data',train=True,
                                           transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root='../../data',train=False,
                                          transform=transforms.ToTensor(),)
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size = batch_size,shuffle=True)

test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size = batch_size,shuffle=False)


#%%
class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_class):
        super(NeuralNet,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,num_class)
        
    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
        
model = NeuralNet(input_size,hidden_layer,num_class)

#%%

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

total_step = len(train_dataloader)
for epoch in range(num_epoch):
    for i,(images,labels) in enumerate(train_dataloader):
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        
        output = model(images)
        loss = criterion(output,labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if(i+1)%100 ==0:
            print('Epoch: [{}/{}], Step:[{}/{}], Loss:{:4f}'.format(epoch+1,num_epoch,i+1,total_step,loss.item()))
        
        
        
        


with torch.no_grad():
    correct = 0
    total = 0
    ##for i,(images,labels) in enumerate(test_dataloader):
    for images,labels in test_dataloader:#in every images, 100 images; in every labels, 100 labels
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        output = model(images)
        _,predicted = torch.max(output.data,1)
        total += images.size(0)#labels.size(0)
        correct += (predicted == labels).sum().item()
    print(correct)
    print(total)
        #if(labels==np.argmax(output,1)):
            #correct = correct +1
        
    print("Accuracy:{:4f}%".format(100 * correct / total))