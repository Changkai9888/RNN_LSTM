import jieba.posseg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import sys
import gensim,fc
torch.manual_seed(2)
bar_init=fc.load_temp('C2020_bar_tick_year_getten_signal')
bar_init=bar_init[14400:9547117];bar_init.reverse()
bar_init=bar_init[140000:590000]
bar_use=[]
batch=2
for i in range(batch):##bar[([2,3,5,2],[1,0,-1,-1])]
    bar_ini=bar_init[i*len(bar_init)//batch:(i+1)*len(bar_init)//batch]
    bar=[[],[]]
    for i in bar_ini:
        bar[0]+=[i[-3]]
        if i[-2]==(1, 1):
            bar[1]+=[3]
        elif i[-2]==(-1, 1):
            bar[1]+=[2]
        elif i[-2]==(1, -1):
            bar[1]+=[1]
        elif i[-2]==(-1, -1):
            bar[1]+=[0]
    bar_use+=[bar]
bar=bar_use
#定义网络结构
class LSTMTagger(torch.nn.Module):
    def __init__(self,embedding_dim,hidden_dim,voacb_size,target_size):
        super(LSTMTagger,self).__init__()
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        self.voacb_size=voacb_size
        self.target_size=target_size
        # 使用Word2Vec预处理一下输入文本
        self.embedding=nn.Embedding(self.voacb_size,self.embedding_dim)
        #  LSTM 以 word_embeddings 作为输入, 输出维度为 hidden_dim 的隐状态值
        self.lstm=nn.LSTM(self.embedding_dim,self.hidden_dim)
        ## 线性层将隐状态空间映射到标注空间
        self.out2tag=nn.Linear(self.hidden_dim,self.target_size)
        self.hidden = self.init_hidden()
    def init_hidden(self):
        # 开始时刻, 没有隐状态
        # 关于维度设置的详情,请参考 Pytorch 文档
        # 各个维度的含义是 (Seguence, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))
    def forward(self,inputs):
        #根据文本的稠密向量训练网络，输出维度为hidden维度
        out,self.hidden=self.lstm(inputs.view(len(inputs),1,-1),self.hidden)
        #做出预测
        tag_space=self.out2tag(out.view(len(inputs),-1))
        tags=F.log_softmax(tag_space,dim=1)
        return tags
model=LSTMTagger(21,10,100,4)#(输入，隐藏层维度，压缩维度，输出维度)
loss_function=nn.NLLLoss()
optimizer=optim.SGD(model.parameters(),lr=0.1)
#看看随机初始化网络的分析结果
for epoch in range(1000):
    # 再说明下, 实际情况下你不会训练300个周期, 此例中我们只是构造了一些假数据
    for p ,t in bar:
        # Step 1. 请记住 Pytorch 会累加梯度
        # 每次训练前需要清空梯度值
        model.zero_grad()
        # 此外还需要清空 LSTM 的隐状态
        # 将其从上个实例的历史中分离出来
        # 重新初始化隐藏层数据，避免受之前运行代码的干扰,如果不重新初始化，会有报错。
        model.hidden = model.init_hidden()
        # Step 2. 准备网络输入, 将其变为词索引的Tensor 类型数据
        sentence_in=torch.tensor(p,dtype=torch.float)
        tags_in=torch.tensor(t,dtype=torch.long)
        # Step 3. 前向传播
        tag_s=model(sentence_in)
        # Step 4. 计算损失和梯度值, 通过调用 optimizer.step() 来更新梯度
        loss=loss_function(tag_s,tags_in)
        loss.backward()
        print(epoch,'Loss:',loss.item())
        optimizer.step()

#看看训练后的结果
with torch.no_grad():
    input_s=torch.tensor(bar[0][0],dtype=torch.float)
    tag_s=model(input_s)

