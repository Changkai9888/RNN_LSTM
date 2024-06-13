from math import exp
import time,torch,datetime,fc,os
import numpy as np
import pickle,random,time,os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
######
#bar=fc.load_temp('bar_C2020_CNN_DQN_指数间隔价格_指数衰减估值')
#整理为LSTM数据
#bar=bar[200010:-20000]
sample0=fc.load_temp('sample0_100万')#sample0=bar[:1000000]#
#sample_test=fc.load_temp('sample_test')#sample_test=bar[3100000:5530000]
######
#定义网络结构
dim=len(sample0[0][-2])
lr=0.1
class LSTM_NET(torch.nn.Module):
    def __init__(self,a_dim,h_dim,t_dim):
        super(LSTM_NET,self).__init__()
        self.h_dim=h_dim
        # 使用Word2Vec预处理一下输入文本
        #  LSTM 以 word_embeddings 作为输入, 输出维度为 hidden_dim 的隐状态值
        self.lstm=nn.LSTM(a_dim,h_dim)
        ## 线性层将隐状态空间映射到标注空间
        self.out2tag=nn.Linear(h_dim,t_dim)
        self.hidden = self.init_hidden()
    def init_hidden(self):
        # 开始时刻, 没有隐状态
        # 各个维度的含义是 (Seguence, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.h_dim),
                torch.zeros(1, 1, self.h_dim))
    def forward(self,inputs):
        #根据文本的稠密向量训练网络，输出维度为hidden维度
        out,self.hidden=self.lstm(inputs.view(len(inputs),1,-1),self.hidden)
        #做出预测
        tag=self.out2tag(out.view(len(inputs),-1))
        return tag
net=LSTM_NET(dim,10,1)#(输入，隐藏层维度，输出维度)
optimizer=optim.SGD(net.parameters(),lr=lr)
loss_function = nn.MSELoss()#设置均方差损失函数
######
input0=[];target0=[]
for i in sample0:
    input0+=[i[-2]];target0+=[[i[-1]]]
input0=torch.tensor(input0,dtype=torch.float)
target0=torch.tensor(target0,dtype=torch.float)
for epoch in range(1000):
    optimizer.zero_grad() 
    net.zero_grad();net.hidden = net.init_hidden()
    output0=net(input0)
    loss=loss_function(output0,target0)
    loss.backward()
    optimizer.step()
    print(epoch,'Loss:',loss.item())
