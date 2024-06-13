path='C:/Quant工坊'
import sys
sys.path.append(path)
from math import exp
import time,torch,datetime,fc,os
import numpy as np
import pickle,random,time,os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
############
class Net(nn.Module): # 网络属于nn.Module类
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv1d(400,10,5)
        self.conv2=nn.Conv1d(400,15,6)
        self.fc1 = nn.Linear(400, 400) # an affine operation: y = Wx + b
        self.fc2 = nn.Linear(400, 40)
        self.fc3 = nn.Linear(40, 1)
    def forward(self, x):
        x=F.pool(F.relu(self.conv1(x)),2)
        x=F.pool(F.relu(self.conv2(x)),2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x
net = Net()
learning_rate=0.02
optimizer=optim.SGD(net.parameters(), lr=learning_rate) #设置随机梯度下降法SGD
############
klines=fc.load_temp('rb000_60s_all')
bar=[]
for i in range(len(klines)-1):
    bar+=[klines.loc[i].close]
############
batch_size=5
epoch_size=10000
############
running_loss = 0.0
net.zero_grad()
optimizer.zero_grad()
for i in range(len(bar)-400-100):
    input0=Variable(torch.tensor(bar[i:i+400]))# 获取输入
    # 梯度置 0
    # 正向传播，反向传播，优化
    outputs=net(inputs)
    output=output.to(torch.float64)
    criterion=nn.MSELoss()#设置：均方差损失函数
    loss=criterion(outputs, labels)
    loss.backward()
optimizer.step()
# 打印状态信息
running_loss += loss.item()


