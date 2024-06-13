import torch,fc,time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
t1=time.time()
torch.manual_seed(2)
# sys.stdout = open('1.log', 'a')
sample0=fc.load_temp('bar_net_0_20wan')
sample_test=fc.load_temp('bar_net_20wan_30wan')
def out_texts(sample0):
    price=[2]
    for i in range(len(sample0)-1):
        price+=[sample0[i+1][1]-sample0[i][1]+2]
    for i in range(len(price)):
        if price[i]>=4:
            price[i]=4
        elif price[i]<=0:
            price[i]=0
    target=[]
    for i in sample0:
        if i[-1]>=2:
            target+=[19]
        elif i[-1]<=-2:
            target+=[0]
        else:
            target+=[int(i[-1]*5+10)]
    texts=[[price,target]]
    return texts
###########
texts=[[[53, 54, 81, 87, 21, 73, 1, 81, 12, 6, 73, 42, 99, 28, 61, 68, 73, 89, 18, 97, 13, 43, 2, 47, 1, 71, 82, 23, \
68, 93, 12, 3, 59, 39, 40, 42, 1, 28, 78, 64, 68, 5, 68, 77, 65, 0, 17, 90, 81, 68, 49, 99, 92, 40, 17, 34, 9, 47, 82, \
38, 88, 51, 76, 1, 94, 30, 9, 25, 50, 68, 60, 99, 96, 67, 9, 33, 68, 91, 63, 1, 15, 8, 20, 68, 65, 11, 41, 62, 27, 70, \
20, 68, 48, 79, 95, 31, 99, 84, 77, 68, 20, 85, 35, 99, 36, 55, 81, 68, 2, 82, 44, 74, 83, 41, 4, 37, 1, 94, 68, 91, \
63, 66, 9, 10, 52, 45, 28, 56, 65, 7, 72, 75, 99, 26, 80, 86, 46, 1, 57, 19, 58, 32, 92, 13, 16, 22, 20, 68, 14, 29, \
69, 98, 24, 1],
        [14, 17, 9, 18, 7, 8, 19, 9, 12, 7, 8, 17, 19, 12, 13, 15, 8, 13, 14, 8, 17, 17, 7, 10, 19, 11, 8, 8, 15, \
18, 12, 7, 14, 2, 2, 17, 19, 12, 8, 17, 15, 4, 15, 8, 8, 19, 17, 17, 9, 15, 8, 19, 2, 2, 17, 0, 16, 10, 8, 12, 13, 8, 4, 19, \
9, 17, 16, 9, 3, 15, 8, 19, 2, 17, 16, 13, 15, 2, 8, 19, 13, 5, 8, 15, 8, 8, 1, 17, 1, 7, 8, 15, 17, 1, 17, 7, 19, 2, 0, 15, \
8, 2, 1, 19, 17, 1, 9, 15, 7, 8, 17, 18, 17, 1, 2, 0, 19, 9, 15, 2, 8, 12, 16, 14, 8, 2, 12, 11, 8, 3, 17, 8, 19, 13, 17, 0, \
17, 19, 6, 17, 3, 8, 2, 17, 12, 5, 8, 15, 2, 8, 6, 8, 17, 19]]]
#####
texts=out_texts(sample0)
texts_test=out_texts(sample_test)
loss_1=[];loss_2=[]
#####
in_dim=5
out_dim=20
def formart_input(inputs):
    return torch.tensor(inputs,dtype=torch.long)
#根据词性表把文本标注输入转换成对应的词汇标注的张量
def formart_tag(inputs):
    return torch.tensor(inputs,dtype=torch.long)
#定义网络结构
class LSTMTagger(torch.nn.Module):
    def __init__(self,embedding_dim,hidden_dim,voacb_size,target_size):
        super(LSTMTagger,self).__init__()
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        self.voacb_size=voacb_size
        self.target_size=target_size
        # 使用Word2Vec预处理一下输入文本voacb_size100--->embedding_dim10
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
        # 预处理文本转成稠密向量
        embeds=self.embedding((inputs))
        #根据文本的稠密向量训练网络
        out,self.hidden=self.lstm(embeds.view(len(inputs),1,-1),self.hidden)
        #做出预测
        tag_space=self.out2tag(out.view(len(inputs),-1))
        tags=F.log_softmax(tag_space,dim=1)
        return tags
net=LSTMTagger(10,10,in_dim,out_dim)
loss_function=nn.NLLLoss()
optimizer=optim.SGD(net.parameters(),lr=0.8)
#看看随机初始化网络的分析结果
'''with torch.no_grad():
    input_s=formart_input(texts[0][0])
    print(input_s)
    tag_s=net(input_s)
    for i in range(tag_s.shape[0]):
        print(tag_s[i])
    # print(tag_s)'''
for epoch in range(4000):
    print('epoch:',epoch)
    # 再说明下, 实际情况下你不会训练300个周期, 此例中我们只是构造了一些假数据
    for p ,t in texts:
        # Step 1. 请记住 Pytorch 会累加梯度
        # 每次训练前需要清空梯度值
        net.zero_grad()
        # 此外还需要清空 LSTM 的隐状态
        # 将其从上个实例的历史中分离出来
        # 重新初始化隐藏层数据，避免受之前运行代码的干扰,如果不重新初始化，会有报错。
        net.hidden = net.init_hidden()
        # Step 2. 准备网络输入, 将其变为词索引的Tensor 类型数据
        sentence_in=formart_input(p)
        tags_in=formart_tag(t)
        # Step 3. 前向传播
        tag_s=net(sentence_in)
        # Step 4. 计算损失和梯度值, 通过调用 optimizer.step() 来更新梯度
        loss=loss_function(tag_s,tags_in)
        loss.backward()
        print('Loss:',loss.item())
        loss_1+=[loss.item()]
        optimizer.step()
    for p ,t in texts_test:
        net.zero_grad()
        net.hidden = net.init_hidden()
        sentence_in=formart_input(p)
        tags_in=formart_tag(t)
        tag_s=net(sentence_in)
        loss=loss_function(tag_s,tags_in)
        print('Loss_test:',loss.item())
        loss_2+=[loss.item()]
#看看训练后的结果
with torch.no_grad():
    input_s=formart_input(texts[0][0])
    tag_s=net(input_s)
    a=[]
    for i in range(tag_s.shape[0]):
        c=0
        while max(tag_s[i])!=tag_s[i][c]:
            c+=1
        a+=[c]
    #print(a)
    #print(tags_in.tolist())
    print(tags_in.tolist()==a)
    fc.plot([tags_in.tolist(),a],k=1)
t1=(time.time()-t1)/3600
print(t1)
#######保存网络参数
fc.save_temp('net_lstm_dict',net.state_dict())
#net.load_state_dict(fc.load_temp('net_dict'))#读取方法
