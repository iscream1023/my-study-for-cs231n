import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchinfo import summary
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
class layer_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1,stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(64*7*7, 256),
            nn.ReLU(),
            nn.Linear(256, 18)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

    def training(dataloader,model,loss_fn,optimizer):
        size = len(dataloader.dataset)
        for batch,(X,y) in enumerate(dataloader):
                X,y = X.to(device), y.to(device)
                #예측
                pred = model(X)
                loss = loss_fn(pred,y)
                #역전파
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if batch%100 == 0:
                    loss = loss.item()
                    current = (batch+1)*len(X)
                    print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    def test(dataloader,model,loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0,0
        with torch.no_grad():
            for X,y in dataloader:
                X,y = X.to(device),y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred,y).item()
                correct+=(pred.argmax(1)==y).type(torch.float).sum().item()
        test_loss/=num_batches
        correct/=size
        print(f"Test error:\n Accuaracy :{(100*correct):>0.1f}%,Avg loss:{test_loss:>8f}\n")
model = layer_NN()
summary(model, input_size=(1, 28, 28))