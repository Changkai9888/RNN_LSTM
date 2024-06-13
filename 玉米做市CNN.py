from math import exp
import time,torch,datetime,fc,os
import numpy as np
import pickle,random,time,os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
##########
bar=fc.load_temp('C2020_bar_tick_year_getten_signal')
bar=bar[14400:9547117]
##########
add=0.01; cost=0.004
hold_time=30
cal_time=240
##########
batch_size=5
epoch_size=10000
learning_rate=0.02
##########
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1   = nn.Linear(21, 21) # an affine operation: y = Wx + b
        self.fc2   = nn.Linear(21, 21)
        self.fc3   = nn.Linear(21, 2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()
criterion = nn.MSELoss()#设置均方差损失函数
##########
def unit(lit):
    lit_out=[]
    for i in lit:
        if max(lit)!=0:
            lit_out+=[i/max(lit)]
        else:
            lit_out+=[1/100]
    return lit_out
##########
def draw_pic(sample_here ,show=0,n=100,if_test=0):
    sample_order=random.sample(range(len(sample_here)),n)
    draw_list=[]
    for i in sample_order:
        input0=[unit(list(sample_here[i][-3]))]
        target=[list(sample_here[i][-2])]
        if if_test==1:
            target=[list(sample_here[i][-1])]
        input0=torch.tensor(input0)
        output =net(input0).detach().numpy()
        draw_list+=[(output[0].tolist(), target[0])]
    plt.figure(figsize=(12, 8))
    for i in draw_list:
        if i[1]==[-1, -1]:
            plt.scatter(i[0][0],i[0][1],s=2, c='#95D0FC', marker='o')
        if i[1]==[-1, 1]:
            plt.scatter(i[0][0],i[0][1],s=2, c='#FF0000', marker='o')
        if i[1]==[1, -1]:
            plt.scatter(i[0][0],i[0][1],s=2, c='#008000', marker='o')
        if i[1]==[1, 1]:
            plt.scatter(i[0][0],i[0][1],s=2, c='#ff9933', marker='o')
    if show==0:
        plt.pause(1)
        plt.clf()
    elif show==1:
        plt.show();plt.close()
    else:
        plt.savefig(r'C:\Quant数据库\图片结果\\'+str(time.time())+show+'.jpg');plt.close()
    #plt.ioff()
    return
##########
sample0=bar[:9000000]
sample_test=bar[:9530000]
iteration=len(sample0)//batch_size
##########
learning_rate_here=learning_rate
for i in range(epoch_size):
    print('epoch:',i)
    sample_order=random.sample(range(len(sample0)),iteration*batch_size)
    count=-1
    right_tot=0;loss_tot=0
    while count<iteration:
        count+=1
        #print('batch:',count)
        input0=[];target=[]
        for k in sample_order[count:count+batch_size]:
            input0+=[unit(list(sample0[k][-3]))]
            target+=[list(sample0[k][-2])]
        input0=Variable(torch.tensor(input0))
        target=Variable(torch.tensor(target))
        net.zero_grad()        
        optimizer = optim.SGD(net.parameters(), lr=learning_rate_here) #设置随机梯度下降法SGD
        optimizer.zero_grad() 
        output = net(input0)
        output=output.to(torch.float64)
        target=target.to(torch.float64)
        loss = criterion(output, target)
        #print('loss:',loss)
        loss.backward()
        optimizer.step()
        loss_tot+=loss
    print('loss_train:',loss_tot/(count+1),'learning_rate',learning_rate_here)
    #if (i<1 and i%3==0) or i%20==0:
        #draw_pic(sample_here=sample0,show='a',n=1000,if_test=0)
        #draw_pic(sample_here=sample_test,show='b',n=1000,if_test=1)

