path='C:/Quant数据库/数据正交化/'
import jieba.posseg
import fc,random,torch
torch.manual_seed(2)
##########
bar=fc.load_temp('bar_tick_year')
bar.reverse()
##########
hand=0;buy_sell=0;p_m=0;count=0;s=0;zhisun=0
##########
price=[]
for i in bar:
    s+=1
    if s%10000==0:
        print(s,buy_sell,count,zhisun)
    price+=[i[1]]
    if len(price)>10000:
        price=price[1:]
    if hand==0:
        a=0;b=0;c=0
        while (a<2000 or b<2000) and c<len(price)-1:
            if i[1]<price[c]:
                a+=1
            if i[1]>price[c]:
                b+=1
            c+=1
        if a>=2000 and b>=2000 and a>b:
            hand=i[0]
            #buy_sell=random.choice([1,-1])
            buy_sell=1
            p_m=buy_sell*i[1]
        if a>=2000 and b>=2000 and a<b:
            hand=i[0]
            #buy_sell=random.choice([1,-1])
            buy_sell=-1
            p_m=buy_sell*i[1]
    if hand!=0:
        if (p_m>0 and i[1]-3>=abs(p_m))or (p_m<0 and i[1]+3<=abs(p_m)):
            hand=0;count+=1
        elif i[0]-hand>200000:
            hand=0;count+=1;zhisun+=abs(i[1]-abs(p_m))

