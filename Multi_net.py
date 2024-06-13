import torch.nn as nn,torch
import torch.nn.functional as F
from 结合律加速 import associa
from 结合律加速 import get_Normalization
from 结合律加速 import f_gather
import time,fc
############
#torch.set_default_dtype(torch.float32)
class Net(nn.Module): # 网络属于nn.Module类
    def __init__(net):
        super(Net, net).__init__()
        net.conv00=nn.Conv1d(2,8,5,stride=5)
        net.conv10=nn.Conv1d(2,8,5,stride=5)
        net.fc11 = nn.Linear(79,128) #平仓
        net.fc12 = nn.Linear(128,64)
        
        net.pi_open = nn.Linear(64, 3)
        #False
        net.multi_q = nn.Parameter(torch.randn(8, 64), requires_grad=False)#multi权重，预训练时True，训练时为False
        net.multi_fond = nn.Parameter(torch.randn(1,8), requires_grad=False)#multi资金权重，预训练时True
        net.multi_f1 = nn.Linear(2*8, 16)#总体决定
        net.multi_f2 = nn.Linear(16, 16)
        net.multi_f3 = nn.Linear(16, 8)
        
        net.exit_price = nn.Linear(64, 1)
        #net.pi_sh = nn.Linear(65,1)
        net.amp=nn.Parameter(torch.tensor(0.))
        net.sh=torch.tensor(0.)
        
        net.sig=nn.Sigmoid()
        net.LeakyReLU=nn.LeakyReLU(0.1)
        net.dropout = nn.Dropout(0.5)
        net.Forced_Open=0
        net.Dropout_Open=0#1,Dropout开启

        net.onehot_00 = nn.Parameter(torch.randn(6, 2))#星期数
        net.onehot_01 = nn.Parameter(torch.randn(12, 2))#小时数
        net.onehot_02 = nn.Parameter(torch.randn(12, 2))#5分钟数
        net.onehot_03 = nn.Parameter(torch.randn(5, 3))#12分钟数
        net.onehot_04 = nn.Parameter(torch.randn(10, 1))#10价格尾数数

        net.hold_last_max=60
        net.exit_func=torch.zeros(net.hold_last_max)#平仓函数
        for i in range(len(net.exit_func)):
            net.exit_func[i]=1-torch.relu(-torch.tensor(0.1*i)+1)
        net.exit_func=torch.flip(net.exit_func,dims=[-1])
        net.device=''

    def multi_pi_open(net,x):
        #multi_q=F.softmax(net.multi_q,dim=0)
        x=x.unsqueeze(1)*net.multi_q#(-1,8,64)
        x=net.pi_open(x.reshape(-1,64)).reshape(-1,8,3)
        return x #-1,16,3
    def multi_f(net, x):#
            y = F.relu(net.multi_f1(x.reshape(-1,16)))
            y = F.relu(net.multi_f2(y))
            y = net.multi_f3(y)
            net.weight  = F.softmax(y,dim=1)
            y=torch.mean(net.weight.unsqueeze(2)*x,dim=1)
            return y
    def conv_cell(net, x):#卷积层特征输出
        x1,x2,x3,x4,x5,base_feature=[x[:,i] for i in range(x.shape[1])]
        #print(x1.shape)
        x1=net.LeakyReLU(net.conv00(x1))
        x2=net.LeakyReLU(net.conv10(x2))
        x3=net.LeakyReLU(net.conv10(x3))
        x4=net.LeakyReLU(net.conv10(x4))
        x5=net.LeakyReLU(net.conv10(x5))
        #x1=F.relu(net.conv01(x1))
        x1 = x1.view(-1, net.num_flat_features(x1))
        x2 = x2.view(-1, net.num_flat_features(x2))
        x3 = x3.view(-1, net.num_flat_features(x3))
        x4 = x4.view(-1, net.num_flat_features(x4))
        x5 = x5.view(-1, net.num_flat_features(x5))
        base_feature = base_feature.view(-1, net.num_flat_features(base_feature))
        x=torch.cat((x1,x2,x3,x4,x5), dim=1)
        y=torch.cat((x,base_feature), dim=1)
        return y
    def onehot_feature(net, x):#总体特征输出
        #x=x.to(net.dtype)
        x0=net.onehot_00[x[:,0]]#-torch.mean(net.onehot_00,dim=0)#星期数
        x1=net.onehot_01[x[:,1]]#-torch.mean(net.onehot_01,dim=0)#小时数
        x2=net.onehot_02[x[:,2]]#-torch.mean(net.onehot_02,dim=0)#5分钟数
        x3=net.onehot_03[x[:,3]]#-torch.mean(net.onehot_03,dim=0)#5分钟数
        mean_4=torch.mean(net.onehot_04,dim=0)
        mean_4[:-1]*=0
        x4=net.onehot_04[x[:,4]]-mean_4#10价格尾数数
        #print(torch.cat((x0,x1,x2,x3,x4),dim=1).shape)
        return torch.cat((x0,x1,x2,x3,x4),dim=1)
    def total_feature(net, x, x_, x_onehot,x_mirr,x_nomirr):#总体特征输出
        'hh,非卷积特征'
        def seq(x,x_onehot, x_mirr,x_nomirr):
            conv_x=net.conv_cell(x)#正向特征
            x=torch.cat((conv_x,x_onehot,x_mirr,x_nomirr), dim=1)
            #x=torch.cat((x_onehot,x_mirr,x_nomirr), dim=1)
            if net.Dropout_Open==1:
                x=net.dropout(x)#dropout
            x = F.relu(net.fc11(x))
            x = F.relu(net.fc12(x))
            return x
        x=seq(x,x_onehot,x_mirr,x_nomirr)
        x_onehot[:,-1]*=-1#尾数价格，特征反转
        x_=seq(x_,x_onehot,-x_mirr,x_nomirr)
        return x, x_
    def forward_mirror_Pi(net,input):#镜像整合到Pi策略输出
        x=input[0];x_=input[1]#卷积多周期数据源，以及镜像数据
        x_mirr=input[2][:,1:]#镜面标量数据源(线性斜率)
        x_nomirr=input[3][:,2:]#非镜面标量数据源(volume，position的裸值等)
        x_onehot=net.onehot_feature(input[4])# onehot 编码数据源（时间，价格尾数，品种信息等）
        y0, y0_=net.total_feature(x, x_, x_onehot,x_mirr,x_nomirr)#正向特征：开仓#0.位置留给品种
        #with torch.no_grad():
        open_short=F.softmax(   net.multi_pi_open(y0_)   ,dim=-1)
        open_long=torch.flip(F.softmax(   net.multi_pi_open(y0)   ,dim=-1), dims=[-1])
        open_0=(open_short[:,:,:2]+open_long[:,:,1:])/2#只接收多空
        net.multi_open_0=open_0
        net.multi_volume=torch.sum(F.softmax(net.multi_fond,dim=1)*torch.min(net.multi_open_0,dim=2)[0],dim=1)
        open_0=net.multi_f(open_0)
        
        #net.sh_price=2+5*torch.sigmoid(net.amp)+5*torch.sigmoid(net.exit_price(y0)+net.exit_price(y0_))
        #net.sh_price=-5+net.amp+5*torch.sigmoid(net.exit_price(y0)+net.exit_price(y0_))
        net.sh_price=net.amp+net.exit_price(y0)+net.exit_price(y0_)
        
        #with torch.no_grad():
        #日内平仓技术：5代表下午三点平仓，7代表晚上11点平仓，4.代表周末平仓
        '''net.inday=1-(  ((input[4][:,1]==5.)+(input[4][:,1]==7.))*(input[4][:,2]==11.)*(input[4][:,3]==4.)  \
                        +(torch.sum(input[4],dim=1)==0.) )*1#全0判断，代表没有数据传入（补0导致）'''
        net.inday=1-(  ((input[4][:,1]==5.)+(input[4][:,1]==7.)*(input[4][:,0]==4.))*(input[4][:,2]==11.)*(input[4][:,3]==4.)  \
                        +(torch.sum(input[4],dim=1)==0.) )*1#全0判断，代表没有数据传入（补0导致）
        return open_0,net.sh_price
    def forward_short(net, input, day_max, mod,if_out_pos_prob=0):#每一个数据
        #mod=0,pos0的arg，给出action概率的抽样结果pos
        #mod=1,保留，可能用来做LSTM串行
        #mod=2,真实序贯抽样决策结果
        ####
        net.hold_last_max=min(net.hold_last_max,day_max)#最大持仓长度
        if mod==1:#LSTM通道
            open_0,exit_price=net.forward_mirror_Pi(input)#转移矩阵
            open_0=open_0.reshape(-1,day_max,*open_0.shape[1:])#卷起来
            exit_price=exit_price.reshape(-1,day_max,*exit_price.shape[1:])#卷起来
            real_price=input[2][:,0].reshape(-1,day_max,1)
            net.inday=net.inday.reshape(-1,day_max)
            ####
            out_pos_prob=torch.zeros(len(open_0),day_max,2).to(net.device)#输出仓位
            pos_list=torch.zeros(len(open_0),net.hold_last_max,2).to(net.device)#开仓列表
            price_list=torch.zeros(len(open_0),net.hold_last_max,2).to(net.device)#止盈止损列表
            for i in range(day_max):
                exit_stop=1-torch.tanh(torch.relu(torch.abs(real_price[:,i]-price_list[:,:,0])-price_list[:,:,1]))
                exit_stop*=net.exit_func[-net.hold_last_max:].unsqueeze(0)
                pos_list*=exit_stop.unsqueeze(2)#平仓
                #fc.plot(pos_list[0])
                pos_list[:,1:]=pos_list[:,:-1].clone()#更新pos_list
                price_list[:,1:]=price_list[:,:-1].clone()#更新price_list
                out_pos_prob[:,i]=(1-torch.sum(pos_list[:,1:],dim=1))*open_0[:,i]#求平仓的开仓量
                pos_list[:,0]=out_pos_prob[:,i];#更新pos_list，更新新开仓
                price_list[:,0,0]=real_price[:,i,0];  price_list[:,0,1]=exit_price[:,i,0]#更新price_list，开仓价，平仓阈值
                '日内平仓技术：5代表下午三点平仓，7代表晚上11点平仓。'
                pos_list*=net.inday[:,i].unsqueeze(1).unsqueeze(2)
                out_pos_prob[:,i]=torch.sum(pos_list,dim=1)
            out_pos_sum=out_pos_prob[:,:,0]*-1+out_pos_prob[:,:,1]#输出净头寸
            out_pos_sum,open_0=out_pos_sum.view(-1),open_0.view(-1,2)#展开输出
            out_pos_sum=out_pos_sum/(torch.max(torch.abs(out_pos_sum)).detach())
            return  out_pos_sum,open_0

    def forward(net, input, day_max,mod,if_out_pos_prob=0):#每一个数据
        if net.device!=input[0].device.type:
            net.device=input[0].device.type
            net.exit_func=net.exit_func.to(net.device)
        net.dtype=input[0].dtype if net.device=='cpu' else torch.float16
        ki=200000
        if len(input[0] )<ki:
            return net.forward_short(input, day_max,mod,if_out_pos_prob)
    def num_flat_features(net, x):
        if len(x.shape)>2:
            size = x.size()[1:]  # all dimensions except the batch dimension
        else:
            size= x.size()
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
#时间反演：
#prob_value_inverse=associa(torch.flip(prob_value,dims=[0]),f=-1,Normalization=1)
#out_pos_prob_inverse=torch.inverse( prob_value_inverse)
