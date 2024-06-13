import fc,TB_functions,numpy as np,pandas as pd,pickle,torch,os,datetime
import torch.nn.functional as F
####
def BN(x_list):
    a=[]
    for x1 in x_list:
        std=torch.std(x1,dim=-1).unsqueeze(-1)
        std[std==0.]+=1e-07
        std=std.expand(x1.shape)#目前：close,V同时做BN
        a+=[(x1-torch.mean(x1,dim=-1).unsqueeze(-1).expand(x1.shape))/std]
    return a
####
def Max_min_list(x_list):
    a=[]
    for x in x_list:
        x=x[:,0,:]
        a+=[((torch.max(x,dim=1)[0]-torch.min(x,dim=1)[0])).unsqueeze(1)]
    return torch.cat(a, dim=1)
####
def diff_list( x_list):
    a=[]
    for x in x_list:
        x=x[:,0,:]
        a+=[(x[:,-1]-x[:,0]).unsqueeze(1)]
    return torch.cat(a, dim=1)
####
def conv_cell(x):
    x=torch.flip(x, dims=[-1])
    lon=5
    x1=x[:,:,:lon*1]#1min,(batch,cv,5)
    x2=torch.cat([torch.mean(x[:,:,i*3:i*3+3],dim=2).unsqueeze(-1) for i in range(lon)],dim=-1)#3min
    x3=torch.cat([torch.mean(x[:,:,i*10:i*10+10],dim=2).unsqueeze(-1) for i in range(lon)],dim=-1)#10min
    x4=torch.cat([torch.mean(x[:,:,i*30:i*30+30],dim=2).unsqueeze(-1) for i in range(lon)],dim=-1)#30min
    x5=torch.cat([torch.mean(x[:,:,i*120:i*120+120],dim=2).unsqueeze(-1) for i in range(lon)],dim=-1)#120min
    Max_min_x=Max_min_list([x1,x2,x3,x4,x5])
    diff_list_x=diff_list([x1,x2,x3,x4,x5])
    base_feature=torch.cat([Max_min_x.unsqueeze(-2),diff_list_x.unsqueeze(-2)],dim=-2)#恰巧组合
    x1,x2,x3,x4,x5=BN([x1,x2,x3,x4,x5])
    x1,x2,x3,x4,x5,base_feature=[i.unsqueeze(1) for i in [x1,x2,x3,x4,x5,base_feature]]
    return torch.cat([x1,x2,x3,x4,x5,base_feature], dim=1)
####
def get_feature(file_name):
    local_path='./data_csv//'
    if file_name+'.pkl' in os.listdir(local_path):
        with open( local_path+file_name+'.pkl','rb') as pickle_file:
            return_to=pickle.load(pickle_file)
        print('本地文件，获取成功！')
        return return_to
    if file_name+'(2)'+'.csv' in os.listdir(local_path):
        df = pd.read_csv(local_path+file_name+'(2)'+'.csv')
    else:
        df = pd.read_csv(local_path+file_name+'.csv')
    if 'time' not in df.keys():
        print('整合datetime')
        for i in range(len(df.datetime)):
            df.loc[i,'time']=datetime.datetime.strptime("{:.6f}".format(df.datetime[i]),"%Y%m%d.%H%M%S")
    "数据的处理"
    c0,v0,=df['close'],df['volume']
    c=np.log(np.array(c0))
    v=np.log(np.array(v0)+1)
    lst_lon=600#样本长度
    star_at=max(lst_lon,345*6)+10#起始点，6日收盘价
    range_list=range(star_at,len(c)-5)
    "注意：输入向量要将镜面向量放置在非镜面向量前面，以便后续处理"
    dtype=torch.float32
    ####原始o以及标签label
    TB_functions.label_open(df)
    TB_functions.label_close(df)
    TB_functions.label_open_log(df)
    TB_functions.label_close_log(df)
    label_open=df['label_open']
    label_close=df['label_close']
    label_open_log=df['label_open_log']
    label_close_log=df['label_close_log']
    labels=np.array([[label_open[i],label_close[i],label_open_log[i],label_close_log[i]] for i in range_list])
    labels=torch.tensor(labels).to(dtype)
    ####卷积数据源，以及镜像卷积数据
    input_0=np.array([[c[i-lst_lon:i],v[i-lst_lon:i]] for i in range_list])
    input_0=torch.tensor(input_0).to(dtype)
    input_0_=torch.clone(input_0)
    input_0_[:,0,:]*=-1
    input_conv=conv_cell(input_0)
    input_conv_neg=conv_cell(input_0_)
    ####镜面标量数据源(线性斜率)，可直接*-1取得标量。不能用最后bar的close，只能用open
    ####要求：对样本平均值须要为零，以免引入绝对单边信号（就是让网络模型无法分辨单个样本的正负号影响）
    TB_functions.open_real(df)#价格尾数编码10    
    open0=df['open_real']
    TB_functions.last_day_high_low(df)#计算过程没有用到当前的close
    close_in_lastday=df['close_in_high_low_last_day']
    close_in_today=df['close_in_high_low_today']
    close_in_bar=df['close_in_high_low_bar']
    TB_functions.LRSlope_n(df,15)#计算过程没有用到当前的close
    TB_functions.LRSlope_n(df,30)#计算过程没有用到当前的close
    TB_functions.LRSlope_n(df,60)#计算过程没有用到当前的close
    TB_functions.LRSlope_n(df,120)#计算过程没有用到当前的close
    TB_functions.LRSlope_n(df,600)#计算过程没有用到当前的close
    LRSlope_15=df['LRSlope_15']
    LRSlope_30=df['LRSlope_30']
    LRSlope_60=df['LRSlope_60']
    LRSlope_120=df['LRSlope_120']
    LRSlope_600=df['LRSlope_600']
    TB_functions.Sharp_ratio_n(df,60)#计算过程没有用到当前的close
    TB_functions.Sharp_ratio_n(df,300)#计算过程没有用到当前的close
    TB_functions.Sharp_ratio_n(df,1000)#计算过程没有用到当前的close
    Sharp_60=df['Sharp_60']
    Sharp_300=df['Sharp_300']
    Sharp_1000=df['Sharp_1000']
    TB_functions.Return_drawdown_ratio_n(df,n=15)
    TB_functions.Return_drawdown_ratio_n(df,n=60)
    TB_functions.Return_drawdown_ratio_n(df,n=120)
    TB_functions.Return_drawdown_ratio_n(df,n=360)
    TB_functions.Return_drawdown_ratio_n(df,n=1000)
    Re_dwdw_15=df['Return_drawdown_15']-df['Return_drawdown_15_rev']
    Re_dwdw_60=df['Return_drawdown_60']-df['Return_drawdown_60_rev']
    Re_dwdw_120=df['Return_drawdown_120']-df['Return_drawdown_120_rev']
    Re_dwdw_360=df['Return_drawdown_360']-df['Return_drawdown_360_rev']
    Re_dwdw_1000=df['Return_drawdown_1000']-df['Return_drawdown_1000_rev']
    TB_functions.last_day_mean_n(df,5)#计算过程没有用到当前的close
    for i in range(1,len(df)):
        df.loc[i,"close_last"]=df['close'][i-1]
    last_5day_gain=(df.close_last-df.last_day_mean_5)/df.last_day_mean_5*100#5日均价增长率
    input_mirr=np.array([[open0[i],close_in_lastday[i], close_in_today[i],close_in_bar[i], LRSlope_15[i],LRSlope_30[i],LRSlope_60[i],LRSlope_120[i],LRSlope_600[i],last_5day_gain[i], \
                                      Sharp_60[i],Sharp_300[i],Sharp_1000[i],\
                                         Re_dwdw_15[i],Re_dwdw_60[i],Re_dwdw_120[i],Re_dwdw_360[i],Re_dwdw_1000[i]\
                          ] for i in range_list])
    input_mirr=torch.tensor(input_mirr).to(dtype)
    ####非镜面标量数据源(volume，position的裸值等)，注意不能用最后bar的vol，因为还没结束最后bar
    open_int=df['position']
    Re_dwdw_15_sum=df['Return_drawdown_15']+df['Return_drawdown_15_rev']
    Re_dwdw_60_sum=df['Return_drawdown_60']+df['Return_drawdown_60_rev']
    Re_dwdw_120_sum=df['Return_drawdown_120']+df['Return_drawdown_120_rev']
    Re_dwdw_360_sum=df['Return_drawdown_360']+df['Return_drawdown_360_rev']
    Re_dwdw_1000_sum=df['Return_drawdown_1000']+df['Return_drawdown_1000_rev']
    input_vol=np.array([[v0[i-1],open_int[i-1],\
                         Re_dwdw_15_sum[i],Re_dwdw_60_sum[i],Re_dwdw_120_sum[i],Re_dwdw_360_sum[i],Re_dwdw_1000_sum[i] ] for i in range_list])
    input_vol=torch.tensor(input_vol).to(dtype)
    #### onehot 编码数据源（时间，价格尾数，品种信息等）
    TB_functions.weekday(df)#星期
    TB_functions.hour_code(df)#时间编码
    TB_functions.minute_code_mod(df,5)#分钟编码5以及余数
    TB_functions.open_real(df)#价格尾数编码10
    weekday=df.weekday
    hour_code=df.hour_code
    minute_code_12=df.minute_mod_5
    minute_code_5=df.minute_remain_5
    open_real_mod_10=np.mod(df.open_real,10)
    input_onehot=np.array([[weekday[i],hour_code[i],minute_code_12[i],minute_code_5[i],open_real_mod_10[i]] for i in range_list])
    input_onehot=torch.tensor(input_onehot).to(torch.int64)
    #F.one_hot(input_onehot[:,2].to(torch.int64)) #如此调用，即可用来编码
    ####
    return_to=labels, input_conv, input_conv_neg, input_mirr,input_vol, input_onehot
    with open( local_path+file_name+'.pkl','wb') as pickle_file:
        pickle.dump(return_to,pickle_file)
    print('从“TB数据下载”中收集，获取成功！')
    df.to_csv(local_path+file_name+'(2)'+'.csv',index=False)
    return return_to
    
if __name__=='__main__':
    a=get_feature('c9888_1m_mini_1w')
#df.to_csv('./data_csv/c9888_1m_mini.csv',index=False)
#df=pd.read_csv('./data_csv/c9888_1m_mini.csv')
