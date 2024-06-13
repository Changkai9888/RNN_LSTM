import torch
import torch.nn as nn
import time
####
bs,T=2,3#批大小，序列长度
input_size,hidden_size=2,4 #输入层大小，隐藏层大小
input =torch.randn(bs,T, input_size)#随机初始化输入特征序列
h_prev=torch.zeros(bs, hidden_size)#初始隐藏状态
####
#step1 调用pytorch_RNN_API
time_a=time.time()
rnn=nn.RNN(input_size,hidden_size,1, batch_first=True)
rnn_output, state_final=rnn(input)#,h_prev.unsqueeze(0))
print('pytorch result:')
print(rnn_output)
print(state_final)
print(time.time()-time_a)
####
#step2 手写RNN
def rnn_forward(input,weight_ih, weight_hh, bias_ih, bias_hh, h_prev):
    bs, T, input_size=input.shape
    h_dim=weight_ih.shape[0]
    h_out=torch.zeros(bs,T,h_dim)#初始化一个输出矩阵

    for t in range(T):
        x=input[:, t, :].unsqueeze(2) #获取当前时刻输入特征，bs*input_size*1
        w_ih_batch=weight_ih.unsqueeze(0).tile(bs,1,1)#bs*h_dim*input_size
        w_hh_batch=weight_hh.unsqueeze(0).tile(bs,1,1)#bs*h_dim*h_dim

        w_time_x=torch.bmm(w_ih_batch,x).squeeze(-1) #bs*h_dim
        w_times_h=torch.bmm(w_hh_batch,h_prev.unsqueeze(2)).squeeze(-1) #bs*h_dim
        h_prev = torch.tanh(w_time_x+bias_ih+w_times_h+bias_hh)

        h_out[:, t, :] = h_prev
    return  h_out, h_prev.unsqueeze(0)
# 验证RNN_forward的正确性
#for k,v in rnn.named_parameters():
    #print(k,v)
time_a=time.time()
custom_rnn_out_put, custom_state_final =rnn_forward(input,rnn.weight_ih_l0, rnn.weight_hh_l0, rnn.bias_ih_l0, rnn.bias_hh_l0, h_prev)
print('rnn_forward output:')
print(custom_rnn_out_put)
print(custom_state_final)
print(time.time()-time_a)
