import get_feature,TB_functions
import fc,torch,numpy as np,fc,time,copy,os,pickle,feather,pandas as pd,EZtree as eztr,random
from LSTM_week_net import Net
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
Trainer_name=os.path.basename(__file__)[:-3]
plot_folder_name='./plot/'+Trainer_name+'/'
####cuda训练数据加载
#file_name_list=['c9888_1m_mini_1w','c9888_1m_mini','c9888_1m','au888_1m','rb888_1m','ni888_1m']
#file_name_list=['c9888_1m_mini_1w','rb888_1m','c9888_1m']#
file_name_list=['m9888.DCE_60s','rb888_1m','c9888.DCE_60s',
                'TA888.CZCE_60s','c9888_1m','au888_1m',
                'a9888.DCE_60s','i9888.DCE_60s']#6,7

cont_data=2
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
    data0=cuda(data0)
    global zoom,cost,cost_real,dtype,labels,labels_test,input,input_test,day_max
    dtype=data0[0].dtype
    if 'i9888' in file_name:
        zoom=1000 ;
        cost_real=0.6/zoom;
    if 'a9888' in file_name:
        zoom=5000 ;
        cost_real=1.26/zoom;
    elif 'c9888' in file_name:
        zoom=5000 ;
        cost_real=1.156/zoom;
    elif 'au888' in file_name:
        zoom=100;
        cost_real=0.04/zoom;
    elif 'rb888' in file_name:
        zoom=5000;
        cost_real=1.5/zoom;
    elif 'm9888' in file_name:
        zoom=5000;
        cost_real=1.195/zoom;
    elif 'TA888' in file_name:
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
        if input_0[4][i][0]<input_0[4][i-1][0]:#周尾数比上一个小
            separate+=[i+1]
    #separate+=[i+1]
    for k in range(len(input_0)):
        input_0[k]=nn.utils.rnn.pad_sequence([input_0[k][separate[i-1]:separate[i]] for i in range(1,len(separate))])
        input_0[k]=input_0[k].permute(1, 0,*range(2, input_0[k].dim())).reshape(-1,*input_0[k].shape[2:])
    labels_0=nn.utils.rnn.pad_sequence([labels_0[separate[i-1]:separate[i]] for i in range(1,len(separate))])
    day_max=len(labels_0)
    labels_0=labels_0.permute(1, 0).contiguous().view(-1)#展为1维
    test_part=int(len(labels_0)*0.2//day_max *day_max)
    labels,labels_test=labels_0[:-test_part],labels_0[-test_part:]
    input=[i[:-test_part] for i in input_0]
    input_test=[i[-test_part:] for i in input_0]
    return labels,labels_test,input,input_test,day_max
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
print("Number of samples: %d" % (len(labels)+len(labels_test)))

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
def test(input,labels,print_str):
    tim_rec2=time.time()
    torch.cuda.empty_cache()#释放显存
    with torch.no_grad(),autocast():
        #net.eval()
        out_pos_sum=net(input,day_max,1)[0]
        Pi_cost=torch.abs(torch.diff(out_pos_sum,prepend=prepend))#try

        right_list_train=out_pos_sum*labels-1*cost*Pi_cost
        #net.train()
    hand_per_day_train=torch.mean(Pi_cost)*345/min_per_bar/2#每天手数实际值
    print(f'test每天手数：{hand_per_day_train}')
    fc.plot(out_pos_sum,save=plot_folder_name+print_str)#+str(cont))
    #fc.plot(output_0_test,save=plot_folder_name+'test_list_测试'+str(cont))
    fc.plot(torch.cumsum(right_list_train,dim=0),save=plot_folder_name+print_str+str(cont))
    #fc.plot(torch.cumsum(output_0_test*labels_test-1*cost*torch.abs(torch.diff(output_0_test, prepend=prepend)),dim=0),save=plot_folder_name+'测试'+str(cont))
    print('检测时间：'+str(time.time()-tim_rec2))
    return right_list_train

    
labels_sh=torch.clone(labels)
sh=cost*1.25
labels_sh[labels_sh>sh]=sh
labels_sh[labels_sh<-sh]=-sh
print('【开始！】')
Train_return_mean=0
while 1:
    cont+=1;#print(cont)
    if cont%10==0 and cont>1:#记录
        #parameter_history+=[copy.deepcopy(net.state_dict())]
        parameter_history+=[{i:net.state_dict()[i].to('cpu') for i in net.state_dict()}]#不会循环占用内存
        torch.cuda.empty_cache()#释放显存
        with torch.no_grad() ,autocast():
            #net.eval()
            
            out_pos_sum_test=net(input_test,day_max,1)[0]
            Pi_cost_test=torch.abs(torch.diff(out_pos_sum_test,prepend=prepend))#try
            
            #net.train()
        right_list=-1*cost_real*Pi_cost+out_pos_sum*labels#_sh
        sharp_ratio= Return(right_list)
        sharp_ratio_list+=[sharp_ratio.tolist()]
        ####
        right_list_test=-1*cost_real*Pi_cost_test+out_pos_sum_test*labels_test
        sharp_ratio_test= Return(right_list_test)
        sharp_ratio_list_test+=[sharp_ratio_test.tolist()]
        ####
        train_lst+=[(torch.sum(out_pos_sum*labels)/torch.sum(Pi_cost)).tolist()]#每手净利润
        test_lst+=[(torch.sum(out_pos_sum_test*labels_test)/torch.sum(Pi_cost_test)).tolist()]#test每手利润
        cost_lst+=[torch.mean(Pi_cost.to(dtype)).tolist()*345/min_per_bar]#平均每天几手
    if cont%10==0 and cont>1:#表现
        print(f'第【{cont}】次，用时:{time.time()-tim_rec}')
        tim_rec=time.time()
        print (f'loss:{loss_lst[-1]}, sharp:{sharp_ratio_list[-1]}')
        print (f'每手利润: train:{train_lst[-1]}, test:{test_lst[-1]}')#loss,每手利润
        print(f'平仓梯度,开仓梯度:{grad_test}')
        print(f'夏普率sharp_max: {max(sharp_ratio_list):.6f}   test每手利润_max: {max(test_lst):.6f}')
        print (f'每天手数:{round(cost_lst[-1],5)}')
        print (f'net.sh_price:{net.sh_price}')
        handle = nvmlDeviceGetHandleByIndex(0);
        meminfo = nvmlDeviceGetMemoryInfo(handle);
        print(f'占用显存：{round(meminfo.used/1073741824,3)}Gbi，显存使用率：{round(meminfo.used/meminfo.total*100,2)}%，显卡温度：{nvmlDeviceGetTemperature(handle, 0)}℃，')
        if cost_lst!=[] and min(cost_lst)<hand_per_day[1]:
            print('hand_control启动！')
            print(net.Forced_Open)
        fc.plot(out_pos_sum,save=plot_folder_name+'pos_list')
        fc.plot(out_pos_sum_test,save=plot_folder_name+'pos_test_list')
        #fc.plot([train_lst,test_lst,[0.0004]*len(train_lst),[0.]*len(train_lst)],k=1,save=plot_folder_name+'right_per_hand')
        fc.plot(np.array([train_lst,test_lst,[cost_real]*len(train_lst)])/cost_real,k=1,save=plot_folder_name+'right_per_hand')
        fc.plot(loss_lst,save=plot_folder_name+'loss_list')
        fc.plot([sharp_ratio_list,sharp_ratio_list_test,[0.]*len(sharp_ratio_list)],k=1,save=plot_folder_name+'sharp_ratio_list')
        fc.plot(cost_lst[:],save=plot_folder_name+'cost_lst_每天手数')
        fc.plot(torch.cumsum( right_list  ,dim=0),save=plot_folder_name+'right_list'+str(cont))#
        fc.plot(torch.cumsum( right_list_test ,dim=0),save=plot_folder_name+'right_list_test'+str(cont))
    if cont%100==0 and cont>1:#检测
        #R_here=test(input,labels,'train训练');test(input_test,labels_test,'test训练');
        #Train_return_mean=Train_return_mean*0.99+R_here*(1-0.99) if 'mean_right_list_train' in dir() else R_here
        pass
    if cont%100==0 and cont>1:#更换数据源
        cont_data=(cont_data+1)%len(file_name_list)
        #training_data(file_name_list[cont_data])
    if cont%5000==0 and cont>1:#备份
        save()
        #parameter_history = []
    #load() if cont==1 else None#加载参数
    #print(net.amp)
    with autocast():
    #if 1==0:
        #out_pos_sum,Pi_cost=net(input,(0,1,0),0)
        #out_pos_sum,out_pos_cost,out_pos_prob,prob_value=net(input,(0,1,0),0,2)
        #prob_value_T,h0=net(input,(0,1,0),0,1)
        #net.train() if net.training==False else None #
        if 1!=1:#依据采样概率
            loss=0
            print('依据采样概率') if cont==1 else None
            here_lon=20000
            begin=random.randint(0,len(input[0])-here_lon)
            input_here=[i[begin:begin+here_lon] for i in input]
            for s in range(40):
                output_0,out_prob=net(input_here,(0,0),2)
                #right_list= out_pos_sum*labels-cost*Prob_cost
                Prob_cost=torch.abs(torch.diff(output_0, prepend=prepend))
                right_list_train=output_0*labels[begin:begin+here_lon]-1*cost*Prob_cost
                #mean_right_list_train=mean_right_list_train*0.99+right_list_train.detach()*(1-0.99) if ('mean_right_list_train' in dir() and cont%10==1) else right_list_train.detach()
                out_prob_log=torch.log(out_prob)
                out_prob_log_cumsum=torch.cumsum(out_prob_log,dim=0)
                #loss=-(sharp_ratio_list[-1]*torch.sum(torch.log(out_prob))+sharp_ratio)/len(out_prob)  #结算夏普率损失函数
                '''Return_here=Return(right_list_train)
                Train_return_mean=Train_return_mean*0.99+Return_here.tolist()*(1-0.99) if cont!=1 else Return_here.tolist()
                loss=-(Return_here-Train_return_mean)*out_prob_log_cumsum[-1]#/len(out_prob)**0.5      #全过程夏普率损失函数'''
                #loss=-torch.sum(torch.cumsum((right_list_train-(mean_right_list_train if 'mean_right_list_train' in dir() else 0)),dim=0)*out_prob_log_cumsum)/(len(right_list_train))**0.5     #盈利损失函数
                a1=torch.cumsum(right_list_train-(mean_right_list_train if 'mean_right_list_train' in dir() else 0),dim=0)
                #a1=torch.cumsum(right_list_train,dim=0)
                k=random.randint(5,345)
                bi=random.randint(15,30)
                loss-=torch.sum((a1[k+bi:]-a1[bi:-k])*(out_prob_log_cumsum[k+bi:]-out_prob_log_cumsum[:-k-bi]))/len(out_prob)**0.5
            loss/=10
            if  cost_lst!=[] and min(cost_lst)<hand_per_day[1]:#等待手数自然下降后，再进行管理。
                hand_per_day_train=torch.mean(Prob_cost.to(data0[0].dtype))*345/2/min_per_bar
                loss+=eztr.hard_slide(hand_per_day_train ,hand_per_day[0],hand_per_day[1])\
                                    *(len(right_list_train))**0.5*torch.sum((Prob_cost-torch.mean(Prob_cost.to(dtype)))*out_prob_log)#0.2,5:每天不超过5手，每周要多于1手
                Forced_Open_admin(hand_per_day_train,hand_per_day[0]*0.5);'强制开仓管理'
        else:#依据混合概率
            print('依据混合概率') if cont==1 else None
            #####
            out_pos_sum,open_0=net(input,day_max,1)
            #print(f'【{cont}】')
            Pi_cost=torch.abs(torch.diff(out_pos_sum,prepend=prepend))#try
            right_list_train=-1*cost*Pi_cost+out_pos_sum*labels#_sh
            loss=torch.tensor(0.).to(net.device)
            loss-=torch.sum(right_list_train)     #总盈利损失函数
            loss-=torch.sum(out_pos_sum*labels)/torch.sum(Pi_cost)*(len(right_list_train))**0.5   #每手净利润
            loss-=Return(right_list_train,mod=2)        #平均夏普率损失函数
            loss-=Return(right_list_train)        #夏普率损失函数
            loss-=eztr.Return_drawdown_ratio(torch.sum(right_list_train.reshape(day_max,-1),dim=0))[0]#每日收益最大回撤比
            if cost_lst!=[] and min(cost_lst)<hand_per_day[1]:
                hand_per_day_train=torch.mean(Pi_cost)*345/2/min_per_bar#每天手数实际值
                open_decide=torch.sum(open_0,dim=1)
                hand_value_log=torch.mean(torch.log(open_decide[open_decide!=0]))#开仓概率，log的目的在于使多的增加较少，少的增加较多。以带来活力
                loss+=eztr.hard_slide(hand_per_day_train ,hand_per_day[0],hand_per_day[1]).detach()*hand_value_log*(len(right_list_train))**0.5#0.2,5:每天不超过5手，每周要多于1手
    loss_lst+=[loss.tolist()]
    if input[0].device.type=='cpu':
        loss.backward()
        optimizer.step()
    elif input[0].device.type=='cuda':
        try:
            scaler.scale(loss).backward()
        except:
            pass
        scaler.step(optimizer)
        scaler.update()
        '''if cont%10==0 and cont>1:
            for param in net.parameters():
                if param.grad==None:
                    continue
                param.grad *= 0.1'''
        ####
    grad_test[0]=[net.exit_price.bias.grad.tolist() if net.exit_price.bias.grad!=None else None]
    grad_test[1]=[net.pi_open.bias.grad.tolist()[0] if net.pi_open.bias.grad!=None else None]
    #print(f'onehot_04: {net.onehot_04.grad[0,0].tolist()}') if cont%100==0 else None
    optimizer.zero_grad()
    #sharp_ratio= torch.sum(right_list)/torch.std(right_list)/(len(output))
    #loss_test_lst+=[(torch.sum(output_test*labels_test)/torch.sum(torch.abs(torch.diff(output_test, prepend=prepend)))).tolist()]#test每手利润
print(f'r2:{r2_score(labels_test.cpu().detach().numpy(),output_test.cpu().detach().numpy())}')
save()
