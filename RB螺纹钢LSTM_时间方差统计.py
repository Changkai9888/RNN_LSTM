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
torch.backends.cudnn.benchmark=True
############
klines=fc.load_temp('rb000_60s_all_cuda')
bar=[];bar_=[]
bar_diff=[[],[]]
length=len(klines)-15#时间序列长度
cod_list=[1,3,5,10,20,30,40,60,90,120,180,240,300,345,400]
for i in range(400,length):
    bar+=[[[],[],[]]]
    bar_+=[[[],[],[]]]
    for k in cod_list:
        bar[-1][0]+=[(klines.loc[i].close-klines.loc[i-k].close)/k**0.5]
        bar_[-1][0]+=[-1*(klines.loc[i].close-klines.loc[i-k].close)/k**0.5]
    for k in range(15):
        bar[-1][-2]+=[klines.loc[i+k].close]
        bar_[-1][-2]+=[-klines.loc[i+k].close]
        bar[-1][-1]+=[klines.loc[i+k].volume]
        bar_[-1][-1]+=[klines.loc[i+k].volume]
    bar_diff[0]+=[klines.loc[i+1].close-klines.loc[i].close]
    bar_diff[1]+=[bar_diff[0][-1]*-1]
####
bar=np.array(bar);bar_=np.array(bar_)
input=Variable(torch.tensor(bar)).to(torch.float32)
input_=Variable(torch.tensor(bar_)).to(torch.float32)
#input.shape=[7700, 2, 300];bar_diff.shape=[1, 7700]
bar_diff=Variable(torch.tensor(bar_diff)).to(torch.float32)
input_test=input;input=input[:-3000]#划分测试集
input_test_=input_;input_=input_[:-3000]#划分测试集
bar_diff_test=bar_diff;bar_diff=bar_diff[:,:-3000]#划分测试集
####
hidden_tensor=torch.zeros(5,1,3)#初始隐藏状态
tensor_reverse=torch.ones(input.shape)
tensor_reverse[:,0,:]*=-1
#step1 调用pytorch_RNN_API
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(3, 64*3,kernel_size=5,stride=5,# 1 input image channel, 6 output channels, 5x5 square convolution kernel
                               groups=3) #groups:通道可分离卷积
        self.conv2 = nn.Conv1d(64*3, 64*3,kernel_size=3,stride=3,# 1 input image channel, 6 output channels, 5x5 square convolution kernel
                               groups=3)
        self.fc1   = nn.Linear(64*3,128) # an affine operation: y = Wx + b
        self.fc2   = nn.Linear(128, 64)
        self.fc3   = nn.Linear(64, 64)
        self.rnn=nn.RNN(64,3,5, batch_first=True,bias=False)
        self.amp_tensor=torch.tensor(10.0,requires_grad=True)
        self.amp_tensor=nn.Parameter(self.amp_tensor)
        #self.amp_tensor.requires_grad=False
    def net_reverse_oneside(self, xi):
        #x.size()=([1, 2, length])
        #for i in range( cov_length,max(x0.size()) ):
            #xi=x0[:,:,i-cov_length:i]
            #print( xi .shape)
        xi = F.relu(self.conv1(xi)) #
        xi = F.relu(self.conv2(xi)) 
        xi = xi.view(-1, self.num_flat_features(xi))
        xi = F.relu(self.fc1(xi))
        xi = F.relu(self.fc2(xi))
        xi = self.fc3(xi)
        y=xi.unsqueeze(0)
        #print( x .shape)
        #(1,length,32)
        return y
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    def forward(self, x0,xi_):
        y=self.net_reverse_oneside(x0)
        y_=self.net_reverse_oneside(xi_)
        y= (y-y_)/2
        y, state_final=self.rnn(y)
        y=y.squeeze(2)
        y=torch.softmax(y,dim=2)
        #y=torch.softmax(y*100,dim=2)
        #y=torch.softmax(y*self.amp_tensor,dim=2)
        return y
RB_cov_rnn= Net()
#RB_cov_rnn.rnn.bias_hh_l0.requires_grad=False;RB_cov_rnn.rnn.bias_hh_l0*=0
#RB_cov_rnn.rnn.bias_ih_l0.requires_grad=False;RB_cov_rnn.rnn.bias_ih_l0*=0
#加载已有训练参数
'''try:
    RB_cov_rnn.load_state_dict(\
        torch.load('RB_cov_rnn_time_var_mini_cuda.pth', \
                   map_location=torch.device('cuda')) )
    RB_cov_rnn.eval()
except:
    pass'''
#加载模型和数据到cuda=1
cuda_on=1
if cuda_on==1:
    RB_cov_rnn.cuda();input=input.cuda();input_=input_.cuda()
    tensor_reverse=tensor_reverse.cuda()
    hidden_tensor=hidden_tensor.cuda();bar_diff=bar_diff.cuda()
    input_test=input_test.cuda();input_test_=input_test_.cuda()
    bar_diff_test=bar_diff_test.cuda()
    RB_cov_rnn.train();print('cuda模式')
#参数数量
total = sum([param.nelement() for param in RB_cov_rnn.parameters()])
print("Number of parameter: %d" % (total))
####
#def main(
learning_rate=0.02
cost_ratio=0
cost_ratio_max=3
file_num=0
grad_list=[1,0.2,0.4,15,0]
grad_list=[0,0,20,0,0]
     #):
optimizer=optim.SGD(RB_cov_rnn.parameters(), lr=learning_rate) #设置随机梯度下降法SGD
right_total=[]
s=0
test_count=3
if file_num!=0:
    RB_cov_rnn.load_state_dict(
        torch.load('net_save\\RB_cov_rnn_time_var_mini_cuda_'+str(file_num)+'.pth',
                   map_location=torch.device('cuda')) )
    RB_cov_rnn.eval();RB_cov_rnn.train()
while cost_ratio<=cost_ratio_max+0.1 and s<=10000:
    s+=1
    optimizer.zero_grad()
    #RB_cov_rnn.zero_grad()
    derta_time=time.time()
    time_forward=time.time()
    rnn_output=RB_cov_rnn(input if s%test_count!=0 else input_test,
                          input_ if s%test_count!=0 else input_test_)
    time_forward=time.time()-time_forward
    #计算利润
    #rnn_output=torch.cat((rnn_output[:,:,0],rnn_output[:,:,2]))
    rnn_output=torch.cat((F.relu(rnn_output[:,:,0]-rnn_output[:,:,2]),F.relu(rnn_output[:,:,2]-rnn_output[:,:,0])))
    hand=torch.cat((rnn_output[:,0].unsqueeze(1),torch.diff(rnn_output)),dim=1)
    hand=torch.abs(hand[0])+torch.abs(hand[1])
    #for i in range(1,len(hand)):
        #hand[i]=torch.abs(rnn_output[0][i]-rnn_output[0][i-1])
    cost=hand*klines.loc[0].close*0.0001*1.3*cost_ratio#设置交易费用
    right=torch.mul(rnn_output,bar_diff if s%test_count!=0 else bar_diff_test)
    right=right[0]+right[1]-cost #收益
    sharp=torch.mean(right)/(torch.std(right)+1)*len(right)**0.5 #夏普率
    right_hand=torch.sum(right)/(torch.sum(hand)+10)#每手收益
    #计算回撤总值：sum_back
    sum_back=torch.sum(F.relu(right*-1))
    sharp_D=torch.mean(right)/(sum_back+1)*len(right)**0.5 #收益回撤比
    loss=torch.cat((-torch.mean(right).unsqueeze(0),#1：平均利润
                    -right_hand.unsqueeze(0),#每手利润
                    -sharp.unsqueeze(0),#夏普
                    -sharp_D.unsqueeze(0),#回撤夏普
            (torch.sum(hand)*0.001+(200**2*0.001)/(torch.sum(hand)+0.00001)).unsqueeze(0) ))#手数100
    gradients=torch.FloatTensor(grad_list).cuda()
    #*torch.abs(1/sharp_D)
    #loss=-torch.sum(right)
    time_back=time.time()
    loss.backward(gradients)
    time_back=time.time()-time_back
    ####cost_ratio__费用/滑点比例扩大判定区
    right_total+=[torch.sum(right).tolist()]
    if s%test_count!=0:
        optimizer.step()
    if len(right_total)>2 and min(right_total[-10:])>10 and sum(abs(RB_cov_rnn.rnn.weight_ih_l0.grad[0][:3]).tolist())<0.08:
        cost_ratio=min(cost_ratio_max,cost_ratio+min(cost_ratio*0.1,0.1)+0.001) if cost_ratio<cost_ratio_max else cost_ratio_max+0.000001
        right_total=[]
    derta_time=time.time()-derta_time   
    ####print/plot__打印画图
    if s%test_count==0:
        right_list=[];right_temp=0;
        for i in right:
            right_temp+=i.tolist();right_list+=[right_temp]
        fc.plot([right_list,[0]*len(input)],k=1,save='C:\\Quant数据库\\图片结果\\RB_RNN\\'+str(s))
        fc.plot((rnn_output[0]-rnn_output[1]).detach().cpu().numpy(),save='C:\\Quant数据库\\图片结果\\RB_RNN\\'+'a_'+str(s))
        print('No.:'+str(s))
        #print('模型计算'+str(time_forward)+'秒')
        #print('反向传播'+str(time_back)+'秒')
        print(str(cost_ratio)+'================')
        print('【手数】'+str(sum(hand).tolist()))
        print('【利润】'+str(torch.sum(right).tolist()))
        print('【夏普率】'+str(sharp.tolist()));
        print('【夏普率D】'+str(sharp_D.tolist()));
        #print(list(RB_cov_rnn.parameters()))
        print('__'+str(RB_cov_rnn.rnn.weight_ih_l0.grad[0][:3]))#梯度
        print(RB_cov_rnn.state_dict()['amp_tensor'])
        print('【Loss】'+str(loss))
        #print(RB_cov_rnn.conv1.weight.grad)
        print('【总时间】'+str(derta_time)+'秒')
        torch.save(RB_cov_rnn.state_dict(),'net_save\\RB_cov_rnn_time_var_mini_cuda_'+str(s)+'.pth')
#torch.save(RB_cov_rnn.state_dict(),'RB_cov_rnn_time_var_mini_cuda.pth')
#RB_cov_rnn.load_state_dict(torch.load('C:\\Quant数据库\\神经网络参数\\RB_cov_rnn_model_weights.pth'));RB_cov_rnn.eval()
#221971
'''main(learning_rate=0.005,
         cost_ratio=0.5,
         cost_ratio_max=5,
         file_num=0,
         grad_list=[1,0.2,0.4,15,1])'''
