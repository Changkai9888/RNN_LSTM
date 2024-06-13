import torch.nn as nn,torch
import torch.nn.functional as F
from 结合律加速 import associa
from 结合律加速 import get_Normalization
from 结合律加速 import f_gather
import time
############
#torch.set_default_dtype(torch.float32)
class Net(nn.Module): # 网络属于nn.Module类
    def __init__(self):
        super(Net, self).__init__()
        self.conv00=nn.Conv1d(2,8,5,stride=5)
        #self.conv01=nn.Conv1d(16,16,5)
        self.conv10=nn.Conv1d(2,8,5,stride=5)
        #self.conv11=nn.Conv1d(32,32,3)
        self.fc11 = nn.Linear(75,128) #平仓
        self.fc12 = nn.Linear(128,64)
        #self.fc1_open = nn.Linear(107,128) #开仓
        #self.fc2_open = nn.Linear(129,64)
        self.pi_open = nn.Linear(64, 3)
        self.pi_exit = nn.Linear(64, 2)
        self.pi_exit_2 = nn.Linear(64, 2)
        #self.pi_sh = nn.Linear(65,1)
        self.amp=torch.tensor(4.6,requires_grad=True)
        self.amp=nn.Parameter(self.amp)
        self.amp2=torch.tensor(0.,requires_grad=True)
        self.amp2=nn.Parameter(self.amp2)
        
        self.sig=nn.Sigmoid()
        self.LeakyReLU=nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.5)
        self.Forced_Open=0
        self.Dropout_Open=0#1时做dropout丢弃
        #self.diamond=torch.tensor([[0., 1., 2.],[1., 0., 1.],[2., 1., 0.]])#开平仓都收费
        self.diamond=torch.tensor([[0., 0., 2.],[2., 0., 2.],[2., 0., 0.]])#只收开仓费
        self.onehot_00 = nn.Parameter(torch.randn(6, 2))#星期数
        self.onehot_01 = nn.Parameter(torch.randn(12, 2))#小时数
        self.onehot_02 = nn.Parameter(torch.randn(12, 2))#5分钟数
        self.onehot_03 = nn.Parameter(torch.randn(5, 3))#12分钟数
        self.onehot_04 = nn.Parameter(torch.randn(10, 1))#10价格尾数数
    def conv_cell(self, x):#卷积层特征输出
        x1,x2,x3,x4,x5,base_feature=[x[:,i] for i in range(x.shape[1])]
        #print(x1.shape)
        x1=self.LeakyReLU(self.conv00(x1))
        x2=self.LeakyReLU(self.conv10(x2))
        x3=self.LeakyReLU(self.conv10(x3))
        x4=self.LeakyReLU(self.conv10(x4))
        x5=self.LeakyReLU(self.conv10(x5))
        #x1=F.relu(self.conv01(x1))
        x1 = x1.view(-1, self.num_flat_features(x1))
        x2 = x2.view(-1, self.num_flat_features(x2))
        x3 = x3.view(-1, self.num_flat_features(x3))
        x4 = x4.view(-1, self.num_flat_features(x4))
        x5 = x5.view(-1, self.num_flat_features(x5))
        base_feature = base_feature.view(-1, self.num_flat_features(base_feature))
        x=torch.cat((x1,x2,x3,x4,x5), dim=1)
        y=torch.cat((x,base_feature), dim=1)
        return y
    def onehot_feature(self, x):#总体特征输出
        #x=x.to(self.dtype)
        x0=self.onehot_00[x[:,0]]#星期数
        x1=self.onehot_01[x[:,1]]#小时数
        x2=self.onehot_02[x[:,2]]#5分钟数
        x3=self.onehot_03[x[:,3]]#5分钟数
        x4=self.onehot_04[x[:,4]]#10价格尾数数
        return torch.cat((x0,x1,x2,x3,x4),dim=1)
    def total_feature(self, x, x_, x_onehot,x_mirr,x_nomirr):#总体特征输出
        'hh,非卷积特征'
        def seq(x,x_onehot, x_mirr,x_nomirr):
            x=self.conv_cell(x)#正向特征
            x=torch.cat((x,x_onehot,x_mirr,x_nomirr), dim=1)
            if self.Dropout_Open==1:
                x=self.dropout(x)#dropout
            x = F.relu(self.fc11(x))
            x = F.relu(self.fc12(x))
            return x
        x=seq(x,x_onehot,x_mirr,x_nomirr)
        x_onehot[:,-1]*=-1#特征反转
        x_=seq(x_,x_onehot,-x_mirr,x_nomirr)
        return x, x_
    def forward_mirror_Pi(self,input):#镜像整合到Pi策略输出
        x=input[0];x_=input[1]#卷积多周期数据源，以及镜像数据
        x_mirr=input[2][:,1:]#镜面标量数据源(线性斜率)
        x_nomirr=input[3][:,2:]#镜面标量数据源(线性斜率)
        x_onehot=self.onehot_feature(input[4])# onehot 编码数据源（时间，价格尾数，品种信息等）
        y0, y0_=self.total_feature(x, x_, x_onehot,x_mirr,x_nomirr)#正向特征：开仓#0.位置留给品种
        #with torch.no_grad():
        open_short=F.softmax(   self.pi_open(y0_)   ,dim=-1)
        open_long=torch.flip(F.softmax(   self.pi_open(y0)   ,dim=-1), dims=[-1])
        open_0=(open_short+open_long)/2
        #with torch.no_grad():
        #y=self.total_feature(x,1.)#正向特征：平仓
        #y_=self.total_feature(x_,1.)#反向特征：平仓
        #with torch.no_grad():pi_exit_2
        #hold__1=torch.cat([F.softmax(self.pi_exit(y0_),dim=-1),torch.zeros(len(x),1).to(self.device)],dim=1)#-1仓位时候
        #hold_1=torch.cat([torch.zeros(len(x),1).to(self.device),torch.flip(F.softmax(self.pi_exit(y0),dim=-1), dims=[-1])],dim=1)#1仓位时候
        
        hold__1=torch.cat([F.softmax(self.pi_exit_2(y0)+self.pi_exit(y0_),dim=-1),torch.zeros(len(x),1).to(self.device)],dim=1)#-1仓位时候
        hold_1=torch.cat([torch.zeros(len(x),1).to(self.device),torch.flip(F.softmax(self.pi_exit_2(y0_)+self.pi_exit(y0),dim=-1), dims=[-1])],dim=1)#1仓位时候
        
        ####
        prob_value=torch.cat([hold__1.unsqueeze(-2),open_0.unsqueeze(-2),hold_1.unsqueeze(-2)],dim=-2)
        #强制平仓Forced Liquidation
        #self.sh=torch.sigmoid(self.amp)*0.999#0.9995代表上限是最多一个月平仓（1分钟bar）
        '''prob_value[:,0]*=self.sh
        prob_value[:,2]*=self.sh
        prob_value[:,0,1]+=(1-self.sh)#.squeeze(-1))
        prob_value[:,2,1]+=(1-self.sh)#.squeeze(-1))'''
        self.sh=torch.tensor(0.9)#固定概率平仓,用于开仓训练
        prob_value[:,0]*=0
        prob_value[:,2]*=0
        prob_value[:,0,0]+=self.sh
        prob_value[:,2,2]+=self.sh
        prob_value[:,0,1]+=(1-self.sh)
        prob_value[:,2,1]+=(1-self.sh)
        '''prob_value[:,0]*=0.999
        prob_value[:,2]*=0.999
        prob_value[:,0,1]+=0.001
        prob_value[:,2,1]+=0.001'''
        #强制观望  Forced balance
        #prob_value[:,1]*=0.999
        #prob_value[:,1,1]+=0.001
        #强制开仓  Forced Open
        '''if self.Forced_Open==1:
            prob_value[:,1]*=0.99
            prob_value[:,1,0]+=0.01
            prob_value[:,1,2]+=0.01'''
        #开平仓协调
        '''k1=torch.min(torch.cat((prob_value[:,0,1].unsqueeze(1),prob_value[:,1,0].unsqueeze(1)),dim=1),dim=1)[0]
        prob_value[:,0,1]-=k1;prob_value[:,0,0]+=k1
        prob_value[:,1,0]-=k1;prob_value[:,1,1]+=k1
        k2=torch.min(torch.cat((prob_value[:,1,2].unsqueeze(1),prob_value[:,2,1].unsqueeze(1)),dim=1),dim=1)[0]
        prob_value[:,2,1]-=k2;prob_value[:,2,2]+=k2
        prob_value[:,1,2]-=k2;prob_value[:,1,1]+=k2'''
        #日内平仓技术：5代表下午三点平仓，7代表晚上11点平仓
        inday=1-(  ((input[4][:,1]==5.)+(input[4][:,1]==7.))*(input[4][:,2]==11.)*(input[4][:,3]==4.)  )*1
        prob_value[:,0]*=inday.unsqueeze(1)
        prob_value[:,2]*=inday.unsqueeze(1)
        prob_value[:,0,1]+=(1-inday)#强制平仓
        prob_value[:,2,1]+=(1-inday)#强制平仓
        prob_value[:,1]*=inday.unsqueeze(1)
        prob_value[:,1,1]+=(1-inday)#强制不许开仓
        #print(inday[29:32]);print(prob_value[26:32])
        #print(inday[254:257]);print(prob_value[254:257])
        return prob_value
    def forward_short(self, input, pos0,mod,if_out_pos_prob=0):#每一个数据
        #mod=0,pos0的arg，给出action概率的抽样结果pos
        #mod=1,保留，可能用来做LSTM串行
        #mod=2,真实序贯抽样决策结果
        ####
        h0=pos0.clone()
        if mod==0:#总仓位，混合概率选择
            #out_pos_prob=torch.zeros(len(x),3).to(self.device)
            prob_value=self.forward_mirror_Pi( input)#转移矩阵
            '''prob_value_T=torch.transpose(prob_value, 1, 2)
            prob_value_T=associa(prob_value_T,f=-1,Normalization=-1)
            #prob_value_T=prob_value_T/(torch.sum(prob_value_T,dim=1).unsqueeze(1).expand(prob_value_T.shape))###概率归一化
            out_pos_prob=prob_value_T@h0'''
            out_pos_prob=h0@associa(prob_value,f=1,Normalization=1)#输出概率向量
            #print(torch.any(torch.isnan(out_pos_prob)));
            diamond=torch.sum(self.diamond.to(self.device).repeat(len(prob_value),1,1)*prob_value,dim=-1)#调仓转移矩阵
            out_pos_cost=torch.cat([pos0.unsqueeze(0),out_pos_prob[:-1]],dim=0)#初始概率向量
            out_pos_cost=torch.sum(out_pos_cost*diamond,dim=1)#交易调仓手数：注意不能通过out_pos_prob的变化直接算调仓手数，因为包含内部的冗余转移，就是平仓和开仓的抵消。
            #print(out_pos_cost);stop;
            out_pos_sum=out_pos_prob[:,0]*-1+out_pos_prob[:,2]#输出净头寸
            if if_out_pos_prob==0:
                return  out_pos_sum,out_pos_cost
            elif if_out_pos_prob==1:#
                return  out_pos_sum,out_pos_cost,prob_value
            elif if_out_pos_prob==2:#测试通道
                return  out_pos_sum,out_pos_cost,out_pos_prob,prob_value
        elif mod==1:#CFR通道
            prob_value=self.forward_mirror_Pi(input)#转移矩阵
            return  prob_value
        elif mod==2:#正常概率选择
            #out_pos_real=torch.zeros(len(x)).to(self.device)
            #out_prob=torch.zeros(len(x)).to(self.device)
            prob_value=self.forward_mirror_Pi(input)
            prob_arg=torch.multinomial(prob_value.view(-1,3), 1, replacement=True).view(-1,3)
            h0_arg=torch.argmax(h0)
            out_pos_real=associa(prob_arg,f=f_gather)[:,h0_arg]
            out_prob=torch.gather(prob_value,1,torch.cat((torch.tensor([[h0_arg]]).to(self.device),out_pos_real[:-1].unsqueeze(1)),dim=0).repeat(1,3).unsqueeze(1)).squeeze(1)
            out_prob=torch.gather(out_prob, 1, out_pos_real.unsqueeze(-1)).squeeze(-1)
            out_pos_real=out_pos_real-1
            return  out_pos_real,out_prob
        elif mod==3:#正常概率选择，拆分100形式
            prob_value=self.forward_mirror_Pi( input)
            out_pos_real=torch.zeros(len(prob_value)).to(self.device)
            out_prob=torch.zeros(len(prob_value)).to(self.device)
            h0_arg=torch.argmax(h0)
            for i in range(len(out_prob)):
                h0_prob=prob_value[i][h0_arg]
                h0_arg=torch.multinomial(h0_prob, 1, replacement=True).squeeze(-1)#随机抽样
                out_prob[i]=h0_prob[h0_arg]
                out_pos_real[i]=h0_arg-1
            return  out_pos_real,out_prob
            
    def forward(self, input, pos0,mod,if_out_pos_prob=0):#每一个数据
        self.device=input[0].device.type
        self.dtype=input[0].dtype if self.device=='cpu' else torch.float16
        ki=200000
        pos0=torch.tensor(pos0).to(self.device)#.to(self.dtype)
        pos0=pos0.to(self.dtype) if mod==0 else pos0
        if len(input[0] )<ki or if_out_pos_prob in(1,2) or mod==2:
            return self.forward_short(input, pos0,mod,if_out_pos_prob)
        input_cut=0
        end_pos_prob=pos0
        out_pos_sum,Pi_cost,out_pos_real,out_prob=[],[],[],[]
        while input_cut<len(input[0]):
            input_mini=[i[input_cut:input_cut+ki] for i in input]
            input_cut+=ki
            if mod==0:#
                out_pos_sum_mini,Pi_cost_mini,end_pos_prob=self.forward_short(input_mini,end_pos_prob,0,1)#本时刻动作概率
                out_pos_sum,Pi_cost=out_pos_sum+[out_pos_sum_mini],Pi_cost+[Pi_cost_mini]
                end_pos_prob=end_pos_prob.tolist()
            if mod==2:
                out_pos_real_mini,out_prob_mini=self.forward_short(input_mini,end_pos_prob,2)
                out_pos_real=out_pos_real+[out_pos_real_mini]
                out_prob=out_prob+[out_prob_mini]
                end_pos_prob=[0,0,0]
                end_pos_prob[int(out_pos_real_mini[-1].tolist()+1)]=1
            if mod==3:
                out_pos_real_mini,out_prob_mini=self.forward_short(input_mini,pos0,3)
                out_pos_real=out_pos_real+[out_pos_real_mini]
                out_prob=out_prob+[out_prob_mini]
        if mod==0:#
            out_pos_sum=torch.cat(out_pos_sum,dim=0)
            Pi_cost=torch.cat(Pi_cost,dim=0)
            if if_out_pos_prob==0:
                return  out_pos_sum,Pi_cost
            else:
                return  out_pos_sum,Pi_cost,end_pos_prob
        if mod==2:#
            out_pos_real=torch.cat(out_pos_real,dim=0)
            out_prob=torch.cat(out_prob,dim=0)
            return out_pos_real,out_prob
        if mod==3:#
            out_pos_real=torch.cat(out_pos_real,dim=0)
            out_prob=torch.cat(out_prob,dim=0)
            return out_pos_real,out_prob
    def num_flat_features(self, x):
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
