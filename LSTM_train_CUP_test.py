import get_feature,TB_functions
import fc,torch,numpy as np,fc,time,copy,os,pickle,feather,pandas as pd,EZtree as eztr,random
from LSTM_net_CUP import Net
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from importlib import reload
from torch.cuda.amp import autocast
import torch.cuda.amp as amp
from pynvml import *
from 结合律加速 import associa
from 结合律加速 import get_Normalization
scaler = amp.GradScaler()
nvmlInit()
'''seed=666
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子'''
####
####模型加载
net = Net()
#####
Trainer_name=os.path.basename(__file__)[:-8]
plot_folder_name='./plot/'+Trainer_name+'/'
####cuda训练数据加载
#file_name_list=['c9888_1m_mini_1w','c9888_1m_mini','c9888_1m','au888_1m','rb888_1m','ni888_1m']
#file_name_list=['c9888_1m_mini_1w','rb888_1m','c9888_1m']#
file_name_list=['m9888.DCE_60s','rb888_1m','c9888.DCE_60s','TA888.CZCE_60s','c9888_1m']
file_name_list=['v9888.DCE_60tb']
#file_name_list=['v9888.DCE_60s2']
cont_data=0
file_name=file_name_list[cont_data]
min_per_bar=1#每个bar是几分钟。（与每天几手的计算相关）
hand_per_day=(0.2,1)#每天几手，(每天不超过5手，每天要多于0.2手)
print(file_name_list)
def cuda(lst):
    a=[]
    for i in lst:
        a+=[i.cuda()]
    net.cuda();#单精度2.8Gbi；半精度
    print(torch.cuda.get_device_name(0))
    return a
def training_data(file_name):
    print('============='+file_name+'=============')
    data0=get_feature.get_feature(file_name)
    '加载cuda'
    #data0=cuda(data0)
    global zoom,cost,cost_real,dtype,labels,input,day_max
    dtype=data0[0].dtype
    if 'v9888' in file_name:
        zoom=5000 ;
        cost_real=1.26/zoom;
    if 'c9888' in file_name:
        zoom=5000 ;
        cost_real=1.156/zoom;
    if 'au888' in file_name:
        zoom=100;
        cost_real=0.04/zoom; 
    if file_name=='rb888_1m':
        zoom=5000;
        cost_real=1.5/zoom;
    if 'm9888' in file_name:
        zoom=5000;
        cost_real=1.195/zoom;
    if 'TA888' in file_name:
        zoom=5000;
        cost_real=1.3*3/zoom;
    '训练cost增加倍率'
    cost=cost_real*1.;
    print(f'缩放倍数：{zoom}；真实费率：{cost_real*zoom}；训练费率倍数：{cost/cost_real}；')
    #cost_real*=0.6;'人为降低交易费用'
    labels_0, input_conv, input_conv_neg, input_mirr, input_vol, input_onehot=data0
    '_';label_open=labels_0[:,0];label_close=labels_0[:,1]
    label_open_log=labels_0[:,2];label_close_log=labels_0[:,3]
    labels_0=label_open/zoom

    #labels_0=label_open_log
    #cost=0.0004
    input_0=[input_conv, input_conv_neg, input_mirr, input_vol, input_onehot]
    separate=[0]
    for i in range(len(input_0[4])):
        if input_0[4][i][1:-1].tolist()==[5,11,4]:#5代表下午三点
            separate+=[i+1]
    #separate+=[i+1]
    for k in range(len(input_0)):
        input_0[k]=nn.utils.rnn.pad_sequence([input_0[k][separate[i-1]:separate[i]] for i in range(1,len(separate))])
        input_0[k]=input_0[k].permute(1, 0,*range(2, input_0[k].dim())).reshape(-1,*input_0[k].shape[2:])
    labels_0=nn.utils.rnn.pad_sequence([labels_0[separate[i-1]:separate[i]] for i in range(1,len(separate))])
    day_max=len(labels_0)
    labels_0=labels_0.permute(1, 0).contiguous().view(-1)#展为1维
    labels=labels_0
    input=[i for i in input_0]
    
    return labels,input,day_max
#########
training_data(file_name)

#labels_train,labels_train_test=labels_train_0[:-test_part-1],labels_train_0[-test_part:]
#learning_rate=0.02
#optimizer=optim.SGD(net.parameters(), lr=0.02) #设置随机梯度下降法SGD
optimizer=torch.optim.Adam(net.parameters(),lr=0.001,betas=(0.9, 0.999),eps=1e-08,weight_decay=0,amsgrad=False)#设置Adam优化算法
prepend=torch.tensor([0]).to(input[0].device.type)
#参数数量
total = sum([param.nelement() for param in net.parameters()])
print("Number of parameter: %d" % (total))
print("Number of samples: %d" % (len(labels)))

####
train_lst=[];test_lst=[];sharp_ratio_list=[];sharp_ratio_list_test=[];mod=0
cost_lst=[];
loss_lst=[]
grad_test=[0,0]
loss=torch.tensor(1000);cont=0;
tim_rec=time.time()
tim=0
parameter_history = []

#cost_test=0.0004
def save():
    with open(plot_folder_name+Trainer_name+'_'+'net_params','wb') as pickle_file:
        pickle.dump((parameter_history,train_lst,test_lst,cont), pickle_file)
    print('很棒！！保存了参数文件，干得不错！！')
    return
def load(s=-1):
    with open(plot_folder_name+Trainer_name+'_'+'net_params','rb') as pickle_file:
            parameter_history_last=pickle.load(pickle_file)[0];print(len(parameter_history_last))
    net.load_state_dict(parameter_history_last[s],strict=False);parameter_history_last=0;print('加载参数！'+str(s))
    return
def Return(right_list_train,mod=0):
    #right_per_hand=torch.sum(out_pos_sum*labels)/torch.sum(Pi_cost)
    #loss=-right_per_hand/abs(right_per_hand.tolist()*50)#((len(right_list))**0.5)#每手利润
    if mod==0:
        a=torch.std(torch.relu(-right_list_train+5e-07),dim=-1)
        a=(a*torch.std(right_list_train,dim=-1))**0.5
        a[a<1e-07]=1e-07
        return  torch.sum(right_list_train,dim=-1)/a/((right_list_train.shape[-1])**0.5)#train下行偏差收益率
    elif mod==1:
        right_list_train-=torch.mean(right_list_train)#.tolist()
        return  Return(right_list_train)
    elif mod==2:
        move=random.randint(1,345)
        right_list_train=right_list_train[:-move]
        ka=random.randint(15,345)
        kb=len(right_list_train)//ka
        right_list_train=torch.sum(right_list_train[:ka*kb].reshape(ka,kb),dim=0)/ka**0.5
        return  Return(right_list_train)
####
def Forced_Open_admin(hand_per_day_train,threshold):
    '强制开仓管理'
    if hand_per_day_train<threshold:
        net.Forced_Open=1
        #print('Forced_Open启动！')
    elif net.Forced_Open==1:
        net.Forced_Open=0
        #print('Forced_Open关闭！')
####
def test(input,labels):
    tim_rec2=time.time()
    torch.cuda.empty_cache()#释放显存
    with torch.no_grad(),autocast():
        out_pos_sum=net(input,day_max,1)[0]
        Pi_cost=torch.abs(torch.diff(out_pos_sum,prepend=prepend))#try

        right_list_train=out_pos_sum*labels-1*cost*Pi_cost
        #net.train()
    hand_per_day_train=torch.mean(Pi_cost)*345/min_per_bar/2#每天手数实际值
    print(f'test每天手数：{hand_per_day_train}')
    fc.plot(out_pos_sum)#+str(cont))
    #fc.plot(output_0_test,save=plot_folder_name+'test_list_测试'+str(cont))
    fc.plot(torch.cumsum(right_list_train,dim=0))
    #fc.plot(torch.cumsum(output_0_test*labels_test-1*cost*torch.abs(torch.diff(output_0_test, prepend=prepend)),dim=0),save=plot_folder_name+'测试'+str(cont))
    print('检测时间：'+str(time.time()-tim_rec2))
    print(f'总利润：{torch.sum(right_list_train).tolist()}')
    return right_list_train

    
labels_sh=torch.clone(labels)
sh=cost*1.25
labels_sh[labels_sh>sh]=sh
labels_sh[labels_sh<-sh]=-sh
print('【开始！】')
Train_return_mean=0
load()
test(input,labels)
