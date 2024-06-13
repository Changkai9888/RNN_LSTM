#对象：引用“04_连日tick数据后复权”结果
#计算价格*交易量的浮动痕迹（指数权重衰减），宽度为20跳价位
path='C:/Quant数据库/数据正交化/'
from math import exp
import fc
import os
import time
import numpy as np
############
bar=fc.load_temp('bar_C2020_all_tick_year')
bar.reverse()
######bar=[[t0,p,v],[t-1,p,v],[t-2,p,v]...]
window=10
cal_bar=30000
hold_bar=7200
hold_bar_test=2*7200
time_decay=6/cal_bar
skip=2
######
exp_list=[]
for i in range(-1,cal_bar+20000):
    exp_list+=[exp(-i*time_decay)]
####
def cal_fp(bar0,bar,bar_1=0):
    t0,p0,v0=bar0
    fp=[0]*(2*window+1)
    if bar_1==0 or int(round(bar_1[1]-p0))>=2:
        for i in bar:
            at=int(round(window-p0+i[1],0))
            if 0<=at<=window*2:
                fp[at]+=i[2]*exp_list[t0-i[0]]
    else:
        p_1=bar_1[1]
        fp=np.array(list(bar_1[-1]))
        fp[10]-v0
        fp=fp*exp_list[0]
        i=bar[-1]
        at=int(round(window-p_1+i[1],0))
        if 0<=at<=window*2:
            fp[at]+=i[2]*exp_list[t0-i[0]]
        if p_1==p0-1:
            fp=np.append(fp[1:],[0])
            for i in bar:
                at=int(round(window-p0+i[1],0))
                if at==window*2:
                    fp[at]+=i[2]*exp_list[t0-i[0]]
        if p_1==p0+1:
            fp=np.append([0],fp[:-1])
            for i in bar:
                at=int(round(window-p0+i[1],0))
                if at==0:
                    fp[at]+=i[2]*exp_list[t0-i[0]]
        fp=fp.tolist()
    return tuple(fp)
####    
for i in range(len(bar)-cal_bar):
    if i%int(len(bar)//100)==0:
        print('1%:'+str(time.time()))
    if i==0:
        bar[i]+=[cal_fp(bar[i][:3],bar[i+1:i+cal_bar+1])]
    else:
        bar[i]+=[cal_fp(bar[i][:3],bar[i+1:i+cal_bar+1],bar[i-1])]
def add_sig(bar, hold_bar,skip):
    for i in range(hold_bar,len(bar)):
        t0,p0,v0=bar[i][:3]
        future_bar=bar[i-hold_bar:i]
        fig=(-1,-1)
        for k in future_bar:
            if k[1]>=p0+skip and fig[1]==-1:
                fig=(fig[0],fig[1]+2)
            if k[1]<=p0-skip and fig[0]==-1:
                fig=(fig[0]+2,fig[1])
        bar[i]+=[fig]
####
add_sig(bar, hold_bar,skip)
add_sig(bar, hold_bar_test,skip)
fc.save_temp('C2020_bar_tick_year_getten_signal',bar)
print(time.ctime())
#bar=bar[14400:9547117]
