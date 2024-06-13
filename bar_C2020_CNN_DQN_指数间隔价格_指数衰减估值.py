#对象：引用“04_连日tick数据后复权”结果
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import fc
######
bar=fc.load_temp('bar_c2020_all_tick_year')
#采集器
collector=[]
for i in range(30):
    collector+=[int(1.5**(i+1))]
#估值器
evaluator=[]
for i in range(600):
    evaluator+=[0.985**i/66.65898161061315]
######
bar_cal=[];hundred=0
for i in range(len(bar)):
    if i<200000 or i>=len(bar)-10000:
        bar_cal+=[bar[i]];continue
    if i%(len(bar)//100)==0:
        hundred+=1
        print(hundred)
    bar_rem=[];value=0
    for k in collector:
        bar_rem+=[bar[i-k][1]]
    for k in range(len(evaluator)):
        value+=(bar[i+k][1]-bar[i][1])*evaluator[k]
    bar_cal+=[bar[i]+[bar_rem]+[value]]
fc.save_temp('bar_C2020_CNN_DQN_指数间隔价格_指数衰减估值',bar_cal)
######
