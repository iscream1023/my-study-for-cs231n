import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.datasets import CIFAR10
from torch.optim import lr_scheduler

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir=r"C:\Users\haggi\PycharmProjects\CNN_project_cn231\.venv\Scripts\runs\CNN_for_drone\act_1")

#data_loot = r""

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

#초기화용 함수
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

#모델 정의
class layer_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding = 1,stride = 1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1,stride = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1,stride = 1)
        self.pool = nn.MaxPool2d(2,2)
        self.linear1 = nn.Linear(128*8*8, 512)
        self.linear2 = nn.Linear(512, 10)
        self.BN1 = nn.BatchNorm2d(32)
        self.BN2 = nn.BatchNorm2d(64)
        self.BN3 = nn.BatchNorm2d(128)
        self.DP = nn.Dropout(p=0.5)
    def forward(self,x):
        x = F.relu(self.BN1(self.conv1(x)))
        x = F.relu(self.BN2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.BN3(self.conv3(x)))
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = F.relu(self.linear1(x))
        x = self.DP(x)
        x = self.linear2(x)
        return x
    def training(self,train_dataloader,model,loss_fn,optimizer,scheduler,epoch,val_dataloader):
        #training
        size = len(train_dataloader.dataset)
        model.train()
        for t in range(epoch):
            print(f"Epoch:{t + 1}\n-----")
            for batch,(X,y) in enumerate(train_dataloader):
                running_loss = 0.0
                X,y = X.to(device), y.to(device)
                optimizer.zero_grad()
                #예측
                pred = model(X)
                loss = loss_fn(pred,y)
                #역전파
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if batch%20 == 0:
                    loss = loss.item()
                    current = (batch + 1) * len(X)
                    print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            avg_train_loss = running_loss / len(train_dataloader)
            scheduler.step()
            #validation
            model.eval()
            correct,total = 0,0
            val_loss = 0.0
            with torch.no_grad():
                for X, y in val_dataloader:
                    X, y = X.to(device), y.to(device)
                    pred = model(X)
                    val_loss += loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                    total += y.size(0)
            avg_val_loss = val_loss/ len(val_dataloader)
            val_acc = correct / total
            print(f"Val:\n Accuaracy :{(val_acc*100):>0.1f}%,Avg loss:{avg_val_loss:>8f}\n")
            writer.add_scalar("Average Train Loss", avg_train_loss, t)
            writer.add_scalar("Average Validation Loss", avg_val_loss, t)
            writer.add_scalar("Validation Acc", val_acc, t)
        writer.close()

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
        writer.add_scalar("test Acc", correct*100)
        print(f"Test error:\n Accuaracy :{(100*correct):>0.1f}%,Avg loss:{test_loss:>8f}\n")

#로컬에서 데이터 불러오기
base = r"C:\Users\haggi\PycharmProjects\cn231_2week\.venv\cifar-10-batches-py"
# 데이터 전처리 옵션
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465) , (0.2023, 0.1994, 0.2010))
])
transform_valtest = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465) , (0.2023, 0.1994, 0.2010))
])
# 데이터 받아오기
full_data = datasets.CIFAR10(
    root = r"C:\Users\haggi\PycharmProjects\cn231_2week\.venv",
    train = True,
    download = False,
    transform=transform_train,
)
test_data = datasets.CIFAR10(
    root = r"C:\Users\haggi\PycharmProjects\cn231_2week\.venv",
    train=False,
    download=False,
    transform=transform_valtest,
)
#파라미터
batch_size = 256
epoch = 50
learning_rate = 1e-4
model = layer_NN().to(device)
model.apply(init_weights)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(),lr = learning_rate,momentum = 0.9,weight_decay = 5e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=60,eta_min=0)

train_size = int(0.6 * len(full_data))
val_size = len(full_data) - train_size
train_data, val_data = random_split(full_data, [train_size, val_size])

train_dataloader = DataLoader(train_data,shuffle = True,batch_size = batch_size)
val_dataloader = DataLoader(val_data,batch_size = batch_size)
test_dataloader = DataLoader(test_data,batch_size = batch_size)
print(torch.cuda.is_available())
print(torch.version.cuda)

layer_NN.training(layer_NN,train_dataloader,model,loss_fn,optimizer,scheduler,epoch,val_dataloader)
layer_NN.test(test_dataloader,model,loss_fn)
print("Done!")