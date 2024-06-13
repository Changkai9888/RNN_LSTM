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
klines=fc.load_temp('rb000_60s_all_cuda')
bar=[]
bar_diff=[]
length=15000#时间序列长度
cov_length=300#卷积长度
for i in range(length-cov_length):
    bar+=[[]]
    bar[-1]+=[klines.loc[i:i+cov_length-1].close]
    bar[-1]+=[klines.loc[i:i+cov_length-1].volume]
    bar_diff+=[klines.loc[i+cov_length].close-klines.loc[i+cov_length-1].close]
bar=np.array(bar)
fc.plot(bar[:,0,0])
input=Variable(torch.tensor(bar)).to(torch.float32)
#input.shape=[7700, 2, 300];bar_diff.shape=[1, 7700]
bar_diff=Variable(torch.tensor(bar_diff)).unsqueeze(0).to(torch.float32)
####
hidden_tensor=torch.zeros(1,1,1)#初始隐藏状态
tensor_reverse=torch.ones(input.shape)
tensor_reverse[:,0,:]*=-1
#input=torch.randn(bs,T, input_size)#随机初始化输入特征序列
####
#step1 调用pytorch_RNN_API
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(2, 64,kernel_size=5,stride=5,# 1 input image channel, 6 output channels, 5x5 square convolution kernel
                               groups=1) #groups:通道可分离卷积
        self.conv2 = nn.Conv1d(64, 64, kernel_size=6,stride=3)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=6,stride=2)
        self.fc1   = nn.Linear(448, 320) # an affine operation: y = Wx + b
        self.fc2   = nn.Linear(320, 80)
        self.fc3   = nn.Linear(80, 32)
        self.rnn=nn.RNN(32,1, batch_first=True)
    def net_reverse_oneside(self, xi):
        #x.size()=([1, 2, length])
        #for i in range( cov_length,max(x0.size()) ):
            #xi=x0[:,:,i-cov_length:i]
            #print( xi .shape)
        xi = F.relu(self.conv1(xi)) # Max pooling over a (2, 2) window
        xi = F.relu(self.conv2(xi))#x = F.max_pool1d(F.relu(self.conv2(x)), 3) # If the size is a square you can only specify a single number
        #print( xi .device)
        xi = F.relu(self.conv3(xi))
        xi = xi.view(-1, self.num_flat_features(xi))
        print( xi .shape)
        xi = F.relu(self.fc1(xi))
        xi = F.relu(self.fc2(xi))
        xi = self.fc3(xi)
        x=xi.unsqueeze(0)
        #print( x .shape)
        #(1,length,32)
        x, state_final=self.rnn(x,hidden_tensor)
        y=x.squeeze(2)
        return y
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    def forward(self, x0):
        xi_=tensor_reverse*x0
        y=self.net_reverse_oneside(x0)
        y_=self.net_reverse_oneside(xi_)
        return (y-y_)/2
RB_cov_rnn= Net()
#加载已有训练参数
RB_cov_rnn.load_state_dict(\
    torch.load('RB_cov_rnn_model_weights_reverside_cuda.pth', \
               map_location=torch.device('cuda')) )
RB_cov_rnn.eval()
#加载模型和数据到cuda=1
cuda_on=1
if cuda_on==1:
    RB_cov_rnn.cuda();input=input.cuda();tensor_reverse=tensor_reverse.cuda()
    hidden_tensor=hidden_tensor.cuda();bar_diff=bar_diff.cuda()
    RB_cov_rnn.train();print('cuda模式')
#参数数量
total = sum([param.nelement() for param in RB_cov_rnn.parameters()])
print("Number of parameter: %d" % (total))
####
learning_rate=0.001
optimizer=optim.SGD(RB_cov_rnn.parameters(), lr=learning_rate) #设置随机梯度下降法SGD
right_total=[];cost_ratio=3
s=0
while cost_ratio<=3.1 and s<=1500:
    s+=1
    optimizer.zero_grad()
    #RB_cov_rnn.zero_grad()
    print('No.:'+str(s))
    derta_time=time.time()
    time_forward=time.time()
    rnn_output=RB_cov_rnn(input)
    print('模型计算'+str(time.time()-time_forward)+'秒')
    #计算利润
    hand=torch.zeros(length-cov_length).cuda()if cuda_on==1 else torch.zeros(length-cov_length)
    hand[0]=torch.abs(rnn_output[0][0])
    for i in range(1,len(hand)):
        hand[i]=torch.abs(rnn_output[0][i]-rnn_output[0][i-1])
    cost=hand*input[0][0][0]*0.0001*1.3*cost_ratio#设置交易费用
    right=torch.mul(rnn_output,bar_diff).squeeze(0)-cost #收益
    sharp=torch.mean(right)/torch.std(right) #夏普率
    loss=-sharp
    time_back=time.time()
    loss.backward()
    print('反向传播'+str(time.time()-time_back)+'秒')
    ####cost_ratio__费用/滑点比例扩大判定区
    right_total+=[torch.sum(right).tolist()]
    optimizer.step()
    if len(right_total)>2 and min(right_total[-10:])>0 and sum(abs(RB_cov_rnn.rnn.weight_ih_l0.grad[0][:3]).tolist())<0.05:
        cost_ratio=min(3,cost_ratio*1.05+0.0001) if cost_ratio<3 else 3.000001
        right_total=[]
    ####print/plot__打印画图
    print(str(cost_ratio)+'================')
    print('【手数】'+str(sum(hand).tolist()))
    print('【利润】'+str(torch.sum(right).tolist()))
    right_list=[];right_temp=0;
    for i in right:
        right_temp+=i.tolist();right_list+=[right_temp]
    if s%20==0:
        fc.plot(right_list,save='C:\\Quant数据库\\图片结果\\RB_RNN\\'+str(s))
        fc.plot(rnn_output[0].detach().cpu().numpy(),save='C:\\Quant数据库\\图片结果\\RB_RNN\\'+'a_'+str(s))
    print('【夏普率】'+str(sharp.tolist()));
    #print(list(RB_cov_rnn.parameters()))
    print('__'+str(RB_cov_rnn.rnn.weight_ih_l0.grad[0][:3]))#梯度
    #print(RB_cov_rnn.conv1.weight.grad)
    print('【总时间】'+str(time.time()-derta_time)+'秒')
torch.save(RB_cov_rnn.state_dict(),
           'RB_cov_rnn_model_weights_reverside_cuda.pth')
#RB_cov_rnn.load_state_dict(torch.load('C:\\Quant数据库\\神经网络参数\\RB_cov_rnn_model_weights.pth'));RB_cov_rnn.eval()
#221971
