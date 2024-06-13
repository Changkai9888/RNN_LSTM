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
        #net.conv01=nn.Conv1d(16,16,5)
        net.conv10=nn.Conv1d(2,8,5,stride=5)
        #net.conv11=nn.Conv1d(32,32,3)
        net.fc11 = nn.Linear(81,128) #平仓
        net.fc12 = nn.Linear(128,64)
        net.fc1_open = nn.Linear(79,128) #开仓
        net.fc2_open = nn.Linear(128,64)
        net.pi_open = nn.Linear(64, 3)
        net.exit_price= nn.Linear(64,1)
        net.exit_price_1 = nn.Linear(64,1)
        #net.pi_sh = nn.Linear(65,1)
        net.amp=nn.Parameter(torch.tensor([10.],requires_grad=True))#宽度乘数
        net.amp2=nn.Parameter(torch.tensor([10.],requires_grad=True))#宽度基础
        net.amp3=nn.Parameter(torch.tensor([3.],requires_grad=True))#倾向
        net.sh=torch.tensor(0.)
        
        net.sig=nn.Sigmoid()
        net.LeakyReLU=nn.LeakyReLU(0.1)
        net.dropout = nn.Dropout(0.5)
        net.Forced_Open=0
        net.Dropout_Open=1#1,Dropout开启
        net.Drop_vect =nn.Parameter(torch.randn(79),requires_grad=True)

        net.onehot_00 = nn.Parameter(torch.randn(6, 1))#星期数
        net.onehot_01 = nn.Parameter(torch.randn(12, 1))#小时数
        net.onehot_02 = nn.Parameter(torch.randn(12, 1))#5分钟数
        net.onehot_03 = nn.Parameter(torch.randn(5, 2))#12分钟数
        net.onehot_04 = nn.Parameter(torch.randn(10, 2))#10价格尾数数

        net.hold_last_max=60
        net.exit_func=torch.zeros(net.hold_last_max)#平仓函数
        for i in range(len(net.exit_func)):
            net.exit_func[i]=1-torch.relu(-torch.tensor(0.2*i)+1)
        net.exit_func=torch.flip(net.exit_func,dims=[-1])
        net.device=''
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
                #x=x*torch.softmax(torch.randn(len(x[0])).to(net.device),dim=0)*len(x[0])
                #x=x*torch.softmax(net.Drop_vect, dim=0)
            return x
        x=seq(x,x_onehot,x_mirr,x_nomirr)
        x_onehot[:,-1]*=-1#尾数价格，特征反转
        x_=seq(x_,x_onehot,-x_mirr,x_nomirr)
        return x, x_
    def open_feature(net,x,x_,fc11,fc12):
        x = F.relu(fc11(x))
        x = F.relu(fc12(x))
        x_ = F.relu(fc11(x_))
        x_ = F.relu(fc12(x_))
        return x, x_
    def forward_mirror_Pi(net,input):#镜像整合到Pi策略输出
        x=input[0];x_=input[1]#卷积多周期数据源，以及镜像数据
        x_mirr=input[2][:,1:]#镜面标量数据源(线性斜率)
        x_nomirr=input[3][:,2:]#非镜面标量数据源(volume，position的裸值等)
        x_onehot=net.onehot_feature(input[4])# onehot 编码数据源（时间，价格尾数，品种信息等）
        y, y_=net.total_feature(x, x_, x_onehot,x_mirr,x_nomirr)#正向特征：开仓#0.位置留给品种
        y1, y1_=net.open_feature(y, y_,net.fc1_open,net.fc2_open)
        #open_0=-open_0[:,0]#只做多
        #print(open_0.shape)
        #net.sh_price=torch.cat([net.amp3*torch.sigmoid(net.exit_price(y0)+net.exit_price(y0_)),#预测价增量，平仓阈值
                                                #+net.amp2+net.amp*torch.sigmoid(net.exit_price_1(y0)+net.exit_price_1(y0_))] ,dim=1)#平仓阈值
        #net.sh_price=torch.cat([net.amp3*torch.tanh(net.exit_price(y1)+net.exit_price(y1_)),#预测价增量，平仓阈值
                                                #+net.amp2+net.amp*torch.sigmoid(torch.abs((net.exit_price_1(y1)+net.exit_price_1(y1_))))] ,dim=1)#平仓阈值
        net.sh_price=torch.cat([net.amp3*torch.tanh(net.exit_price(y1)+net.exit_price(y1_)),#预测价增量，
                                                +net.amp2+net.amp*torch.sigmoid(torch.abs((net.exit_price_1(y1)+net.exit_price_1(y1_))))] ,dim=1)#平仓阈值
        #net.sh_price[:,0]*=0.
        #net.sh_price=net.exit_price(y0)+net.exit_price(y0_)
        open_y=torch.cat([y,net.sh_price],dim=1 )
        open_y_=torch.cat([y_,net.sh_price*torch.tensor([-1.,1.]).to(net.device) ],dim=1 )
        y0, y0_=net.open_feature(open_y, open_y_,net.fc11,net.fc12)#平仓的想法灌输的开仓中
        
        open_short=F.softmax(   net.pi_open(y0_)   ,dim=-1)#平仓的概率体现在过程中
        open_long=torch.flip(F.softmax(   net.pi_open(y0)   ,dim=-1), dims=[-1])
        open_0=(open_short[:,:2]+open_long[:,1:])/2#只接收多空
        open_0=-open_0[:,0]+open_0[:,1]
        
        #日内平仓技术：5代表下午三点平仓，7代表晚上11点平仓，4.代表周末平仓
        '''net.inday=1-(  ((input[4][:,1]==5.)+(input[4][:,1]==7.))*(input[4][:,2]==11.)*(input[4][:,3]==4.)  \
                        +(torch.sum(input[4],dim=1)==0.) )*1#全0判断，代表没有数据传入（补0导致）'''
        net.inday=1-(  ((input[4][:,1]==5.)*(input[4][:,0]!=4.)+(input[4][:,1]==7.)*(input[4][:,0]==4.))*(input[4][:,2]==11.)*(input[4][:,3]==4.)  \
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
            exit_price=exit_price.reshape(-1,day_max,*exit_price.shape[1:])#卷起来，预测价增量，平仓阈值
            real_price=input[2][:,0].reshape(-1,day_max,1)#真实价格
            net.inday=net.inday.reshape(-1,day_max)
            ####
            out_pos_prob=torch.zeros(len(open_0),day_max).to(net.device)#输出仓位
            pos_list=torch.zeros(len(open_0),net.hold_last_max).to(net.device)#开仓列表
            price_list=torch.zeros(len(open_0),net.hold_last_max,2).to(net.device)#止盈止损列表，预测价，平仓阈值记录
            for i in range(day_max-1):#-1表示最后一轮，尾部平仓处理'
                #print(open_0[:,i])
                
                exit_stop=1-torch.tanh(torch.relu(torch.abs(real_price[:,i]-price_list[:,:,0])-price_list[:,:,1]))#价格边界
                exit_stop*=net.exit_func[-net.hold_last_max:].unsqueeze(0)#整合最大尺寸时间包络
                pos_list*=exit_stop#平仓
                
                #pos_list[:,1:]=pos_list[:,:-1].clone()#更新pos_list
                #price_list[:,1:]=price_list[:,:-1].clone()#更新price_list
                pos_list_last=pos_list[:,:-1].clone()#更新pos_list
                price_list_last=price_list[:,:-1].clone()#更新price_list
                #print(1-torch.sum(torch.abs(pos_list[:,1:]),dim=1))
                #pos_list[:,0]=(1-torch.abs(torch.sum(pos_list[:,1:],dim=1)))*open_0[:,i]#求平仓的开仓量，更新pos_list，更新新开仓
                sign_pos=torch.sign(torch.sum(pos_list[:,1:],dim=1))
                sign_open=torch.sign(open_0[:,i])
                sign_pos[sign_pos==0]=sign_open[sign_pos==0]
                'pos_list_head'
                #pos_list_head=F.relu((1-torch.sum(torch.abs(pos_list_last),dim=1))*open_0[:,i]*sign_pos)*sign_pos#求平仓的开仓量，更新pos_list，更新新开仓
                pos_list_head=F.relu((1-torch.sum(torch.abs(pos_list_last),dim=1)))*open_0[:,i]#求平仓的开仓量，更新pos_list，更新新开仓
                #have=torch.sum(torch.abs(pos_list),dim=1)
                #have[have==0]=1
                #pos_list_head=F.relu((have-torch.sum(torch.abs(pos_list_last),dim=1))*open_0[:,i]*sign_pos)*sign_pos#求平仓的开仓量，更新pos_list，更新新开仓
                pos_list=torch.cat([pos_list_head.unsqueeze(1),pos_list_last],dim=1)
                #print(pos_list[0])
                #print(exit_price[:,i,1])
                #price_list[:,0,0]=real_price[:,i,0]+exit_price[:,i,0]*torch.sign(pos_list[:,0]);#更新price_list，预测价
                #price_list[:,0,1]=exit_price[:,i,1]#更新price_list，平仓阈值
                #print(price_list.shape)
                #price_list_head=torch.cat([(real_price[:,i,0]+exit_price[:,i,0]*torch.tanh(100*pos_list_head)).unsqueeze(1),exit_price[:,i,1].unsqueeze(1)],dim=1)#偏移价格，价格宽度
                price_list_head=torch.cat([(real_price[:,i,0]+exit_price[:,i,0]).unsqueeze(1),exit_price[:,i,1].unsqueeze(1)],dim=1)#偏移价格，价格宽度
                #print(price_list_head.shape);print(price_list_last.shape)
                price_list=torch.cat([price_list_head.unsqueeze(1),price_list_last],dim=1)
                
                '日内平仓技术：5代表下午三点平仓，7代表晚上11点平仓。'
                pos_list*=net.inday[:,i].unsqueeze(1)
                out_pos_prob[:,i]=torch.sum(pos_list,dim=1)
                
                if 1==1:
                    net.out_pos_prob=out_pos_prob
                    #print(torch.min((out_pos_prob)))
                    #if torch.min((out_pos_prob))<0 or  torch.min(exit_stop)<0:# or torch.max(torch.sum(torch.abs(pos_list_last),dim=1))>1:#超过最大手数
                    net.pos_list=pos_list;
                    net.pos_list_head=pos_list_head
                    net.real_price=real_price
                    net.pos_list_last=pos_list_last
                    net.exit_stop=exit_stop
                    net.pos_list_head=pos_list_head

                #print(torch.sum(pos_list,dim=1).shape)
                #fc.plot([pos_list[3],out_pos_prob[3]],k=1)
                #print(pos_list[0,:100])
                #fc.plot([pos_list[0],out_pos_prob[0],exit_stop[0],real_price[0,:,0]/100-47],k=1) if i>100 else None
            out_pos_sum=out_pos_prob#输出净头寸
            #fc.plot(pos_list[0])#
            #fc.plot(exit_stop[0])#
            #fc.plot(out_pos_sum[10])
            out_pos_sum,open_0=out_pos_sum.view(-1),open_0.view(-1)#展开输出
            #net.out_pos_prob=out_pos_prob
            if torch.max(torch.abs(out_pos_sum))>2:
                stop;
            #print(out_pos_sum.shape)
            #out_pos_sum=out_pos_sum/torch.max(torch.abs(out_pos_sum))
            return  out_pos_sum,open_0    
    def forward(net, input, day_max,mod,if_out_pos_prob=0):#每一个数据
        if net.device!=input[0].device.type:
            net.device=input[0].device.type
            net.exit_func=net.exit_func.to(net.device)
        net.dtype=input[0].dtype if net.device=='cpu' else torch.float16
        ki=300000
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
