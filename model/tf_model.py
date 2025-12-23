from typing import Union, Callable, Tuple, Any, Optional, Dict
import config
import math
import copy
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T
from torch.utils.hooks import RemovableHandle

DEVICE=config.device
'''
将输入的离散词索引转换为连续的向量表示
例如,将词汇表的第5个词映射为512维向量
'''
class Embeddings(nn.Module):
    def __init__(self,d_model,vocab):
        #初始化方法,出入模型的维度,和词表大小
        super(Embeddings,self).__init__()
        #Embedding层,将词表大小映射为d_model维的向量
        self.lut=nn.Embedding(vocab,d_model)
        self.d_model=d_model

    def forward(self,x):
        #返回x对应的embedding矩阵（需要乘以math.sqrt(d_model)）
        #为了保持词向量的方差,使其适应后续层的训练
        return self.lut(x)*math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000):
        super(PositionalEncoding,self).__init__()
        self.dropout=nn.Dropout(p=dropout)

        #初始化一个size为max_len(设定的最大长度)*embedding维度 的全零矩阵
        #来存放所有小与这个长度位置的对应positional embedding
        pe=torch.zeros(max_len,d_model,device=DEVICE)
        #生成一个位置下标的tensor矩阵,每一行都是一个位置下标
        position=torch.arange(0,max_len,device=DEVICE).unsqueeze(1)
        #这里幂运算太多,用exp和log来转换实现公式中的pos下面要除的分母,由于是分母,注意要带负号
        div_term=torch.exp(torch.arange(0.,d_model,2,device=DEVICE)*-(math.log(10000.0)/d_model))

        #根据公式,计算各个位置在各个embedding维度上的位置纹理值,存放到pe矩阵中
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)

        #加一个维度,使pe维度变为:1*max_len*embedding维度
        #方便后续与一个batch句子的所有词的embedding相加
        pe=pe.unsqueeze(0)
        #将pe矩阵以持久的buffer状态存下,(不会作为需要训练的参数)
        self.register_buffer('pe',pe)

    def forward(self,x):
        #将一个batch的句子所有词embedding层与构建好的positional embedding相加
        #(这里按照该批次数据的最大句子长度来取对应需要的那些positional embedding值
        x=x+Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return self.dropout(x)

'''Transformer模型的核心注意力机制'''
def attention(query,key,value,mask=None,dropout=None):
    #将query矩阵的最后一个维度值作为d_k
    d_k=query.size(-1)
    #将key的最后两个维度互换(转置),才能与query相乘,乘完了还要除以d_k开根号
    scores=torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    '''
        Encoder:主要是用padding mask处理不等长序列
        Decoder:同时使用padding mask和sequence mask
        padding mask处理填充部分
        sequence mask防止看到未来信息
        
    '''
    #如果存在要进行mask的内容,则将那些记为0的部分替换成负无穷
    if mask is not None:
        scores=scores.masked_fill(mask==0, -1e9)

    #将mask后面的attention矩阵按照最后一个维度
    p_attn=F.softmax(scores,dim=-1)
    if dropout is not None:
        p_attn=dropout(p_attn)
    #最后返回注意力矩阵跟value的乘积,以及注意力矩阵
    return torch.matmul(p_attn,value),p_attn
    #注意力矩阵跟value的乘积为L*d_k,注意力矩阵为L*L,L为序列长度

class MultiHeadedAttention(nn.Module):
    def __init__(self,h,d_model,dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        #保证可以整除
        assert d_model%h==0
        #得到一个head的attention表示维度
        self.d_k=d_model//h
        self.h = h
        #定义4个全连接函数,供后续作为WQ,WK,WV矩阵和最后h个多头注意力矩阵concat之后进行变换的矩阵
        self.linears=clones(nn.Linear(d_model,d_model),4)
        self.attn=None
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,query,key,value,mask=None):
        if mask is not None:
            mask=mask.unsqueeze(1)
        #query的第一个维度值为batch_size
        nbatches=query.size(0)
        #将embedding层乘以WQ,WK,WV矩阵
        #并将结果拆成h块,然后将第2个和第3个维度值互换
        '''具体操作步骤如下：
        线性变换：
        l(x)：使用MultiHeadedAttention类中定义的线性层对输入进行变换
        self.linears包含了4个线性层，前3个分别用于变换query、key和value
        重塑张量：
        .view(nbatches, -1, self.h, self.d_k)：将线性变换后的张量重新组织为4维形式
        nbatches：批次大小
        -1：序列长度（由系统自动推断）
        self.h：注意力头的数量
        self.d_k：每个注意力头的维度
        维度转置：
        .transpose(1, 2)：交换第1维和第2维
        这样做的目的是让注意力头维度紧邻在批次维度之后，便于并行计算
        举个例子，假设输入的维度是(batch_size, seq_len, d_model)：
        经过线性变换后仍然是(batch_size, seq_len, d_model)
        经过view后变成(batch_size, seq_len, h, d_k)（因为d_model = h * d_k）
        ###经过transpose后变成(batch_size, h, seq_len, d_k)###
        这样变换后，每个注意力头可以独立地在自己的维度上进行计算，实现了真正的多头并行处理。
        '''

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        #输出(batch_size, h, seq_len, d_k)
        #调用上述定义的attention函数计算得到h个注意力矩阵跟value的乘积,以及注意力矩阵
        x,self.attn=attention(query,key,value,mask=mask,dropout=self.dropout)
        # 将h个多头注意力矩阵concat起来（注意要先把h变回到第三维的位置）
        x=x.transpose(1,2).contiguous().view(nbatches,-1,self.h*self.d_k)
        '''具体步骤分解：
        x.transpose(1,2)：
            将张量的第1维和第2维进行交换
            假设此时x的维度是(batch_size, h, seq_len, d_k)（经过注意力计算后的结果）
            转置后变成(batch_size, seq_len, h, d_k)
        .contiguous()：
            确保张量在内存中是连续存储的
            在进行维度转置等操作后，张量在内存中可能不是连续存储的
            调用contiguous()可以确保后续的view操作能够正常执行
        .view(nbatches, -1, self.h*self.d_k)：
            将4维张量重新reshape为3维
        nbatches：批次大小，保持不变
        -1：序列长度维度，由系统自动推断
        self.h*self.d_k：将多头的维度合并，恢复到原始的模型维度d_model（因为d_model = h * d_k）
        最终维度为(batch_size, seq_len, d_model)
        整个操作的目的：
            将多个注意力头的输出结果合并回单一的表示空间
            恢复到与输入相同的维度格式，便于后续层的处理
            为下一步通过最后的线性变换做准备
        这是多头注意力机制中非常重要的一步，它将并行计算的多个头的结果整合起来，形成最终的输出。'''
        # 使用self.linears中构造的最后一个全连接函数来存放变换后的矩阵进行返回
        return self.linears[-1](x)
        #将前面拼接后的矩阵x大小为[seq_len*h,d_model],通过线性变换变为[seq_len,d_model]最后输出,和输入的矩阵一样

'''定义一个层归一化的类'''
class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        super(LayerNorm,self).__init__()
        # 初始化α为全1, 而β为全0
        self.a_2=nn.Parameter(torch.ones(features))
        self.b_2=nn.Parameter(torch.zeros(features))
        #平滑项
        self.eps=eps

    def forward(self,x):
        # 这是一个前向传播函数，用于执行Layer Normalization操作
        # 输入x是神经网络层的输出
        # 按最后一个维度计算均值和方差
        # keepdim=True确保输出的维度与输入相
        mean=x.mean(-1,keepdim=True)
        std=x.std(-1,keepdim=True)

        #返回layerNorm的结果,
        # Layer Norm公式: y = a * (x - mean) / sqrt(std^2 + eps) + b
        return self.a_2*(x-mean)/torch.sqrt(std ** 2+self.eps)+self.b_2

'''前馈神经网络的实现'''
class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        """
        位置前馈神经网络初始化函数
        参数:
            d_model: 模型的输入维度
            d_ff: 前馈神经网络中间层的维度
            dropout: dropout概率，默认为0.1
        """
        super(PositionwiseFeedForward,self).__init__()
        self.w_1=nn.Linear(d_model,d_ff)    # 第一个线性层，将维度从d_model扩展到d_ff
        self.w_2=nn.Linear(d_ff,d_model)    # 第二个线性层，将维度从d_ff压缩回d_model
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))  # 先通过第一个线性层，然后应用ReLU激活函数，


"""
SublayerConnection的作用就是把Multi-Head Attention和Feed Forward层连在一起
"""
class SublayerConnection(nn.Module):
    """
    SublayerConnection的作用就是把Multi-Head Attention和Feed Forward层
    连在一起只不过每一层输出之后都要先做Layer Norm再残差连接
    sublayer是lambda函数
    """
    def __init__(self,size,dropout):
        super(SublayerConnection,self).__init__()
        self.norm=LayerNorm(size)
        self.dropout=nn.Dropout(dropout)

    """
    sublayer参数是SublayerConnection类中的核心组件，
    它代表了Transformer中的具体处理层，通过这种设计实现了代码的模块化和灵活性。
    """
    def forward(self,x,sublayer):
        #返回layerNorm和残差连接的结果
        return x+self.dropout(sublayer(self.norm(x)))

def clones(module,N):
    '''克隆型模块,克隆的模型块参数不共享'''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class EncoderLayer(nn.Module):
    def __init__(self,size,self_attn,feed_forward,dropout):
        super(EncoderLayer,self).__init__()
        self.self_attn=self_attn
        self.feed_forward=feed_forward
        # SublayerConnection的作用就是把multi和ffn连在一起
        # 只不过每一层输出之后都要先做Layer Norm再残差连接
        self.sublayer=clones(SublayerConnection(size,dropout),2)
        self.size=size

    def forward(self,x,mask):
        #将embedding层进行multi head attention
        x=self.sublayer[0](x,lambda x:self.self_attn(x,x,x,mask))
        #注意到attn得到的结果x直接作为下一层的输入
        return self.sublayer[1](x,self.feed_forward)

class Encoder(nn.Module):
    #layer=EncoderLayer
    #N=6
    def __init__(self,layer,N):
        super(Encoder,self).__init__()
        #复制N个EncoderLayer
        self.layers=clones(layer,N)
        #layerNorm
        self.norm=LayerNorm(layer.size)

    def forward(self,x,mask):
        '''使用循环连续Encode N次
            这里的EncoderLayer会接收一个对于输入的attention mask处理
        '''
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self,size,self_attn,src_attn,feed_forward,dropout):
        super(DecoderLayer,self).__init__()
        self.size=size
        self.self_attn = self_attn
        #与Encoder传入的Context进行Attention
        self.src_attn=src_attn
        self.feed_forward=feed_forward
        self.sublayer=clones(SublayerConnection(size,dropout),3)

    def forward(self,x,memory,src_mask,tgt_mask):
        #用m来存放encoder的最终hidden表示结果
        m=memory

        #Self_Attention:注意self-attention的q,k,v均为decoder hidden
        x=self.sublayer[0](x,lambda x:self.self_attn(x,x,x,tgt_mask))
        #Context-attention:注意context-attention的q为decoder hidden,而k和v为encoder hidden
        x=self.sublayer[1](x, lambda x:self.src_attn(x,m,m,src_mask))
        return self.sublayer[2](x,self.feed_forward)

class Decoder(nn.Module):
    def __init__(self,layer,N):
        super(Decoder,self).__init__()
        #复制N个encoder layer
        self.layers=clones(layer,N)
        #layer Norm
        self.norm=LayerNorm(layer.size)

    def forward(self,x,memory,src_mask,tgt_mask):
        '''
        使用循环连续的decode N次,这里为6次
        这里的DecoderLayer会接受一个对于输入的attention mask处理
        和一个对输出的attention mask+ subsequent mask处理
        '''
        for layer in self.layers:
            x=layer(x,memory,src_mask,tgt_mask)
        return self.norm(x)

class Generator(nn.Module):
    #vocab:tgt_vocab
    def __init__(self,d_model,vocab):
        super(Generator,self).__init__()
        #decoder后的结果,先进入一个全连接层变为词典大小的向量
        self.proj=nn.Linear(d_model,vocab)
    def forward(self,x):
        #返回未归一化的logits; 由损失函数内部完成softmax处理
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(self,encoder,decoder,src_embed,tgt_embed,generator):
        super(Transformer,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=src_embed
        self.tgt_embed=tgt_embed
        self.generator=generator
    def encode(self,src,src_mask):
        return self.encoder(self.src_embed(src),src_mask)
    def decode(self,memory,src_mask,tgt,tgt_mask):
        return self.decoder(self.tgt_embed(tgt),memory,src_mask,tgt_mask)
    def forward(self,src,tgt,src_mask,tgt_mask):
        #encoder的结果做为decoder的memory参数传入,进行decode
        return self.decode(self.encode(src,src_mask),src_mask,tgt,tgt_mask)

def make_model(src_vocab,tgt_vocab,N=6,d_model=512,d_ff=2048,h=8,dropout=0.1):
    c=copy.deepcopy
    #实例化Attention图像
    attn=MultiHeadedAttention(h,d_model).to(DEVICE)
    #实例化的FeedForward图像
    ff=PositionwiseFeedForward(d_model,d_ff,dropout).to(DEVICE)
    #实例化PositionalEncoding对象
    position=PositionalEncoding(d_model,dropout).to(DEVICE)

    #实例化Transformer模型对象
    model=Transformer(
        Encoder(EncoderLayer(d_model,c(attn),c(ff),dropout).to(DEVICE),N).to(DEVICE),
        Decoder(DecoderLayer(d_model,c(attn),c(attn),c(ff),dropout).to(DEVICE),N).to(DEVICE),
        nn.Sequential(Embeddings(d_model,src_vocab).to(DEVICE),c(position)),
        nn.Sequential(Embeddings(d_model,tgt_vocab,).to(DEVICE),c(position)),
        Generator(d_model,tgt_vocab)).to(DEVICE)

    #初始化模型参数
    #遍历模型中的所有参数
    for p in model.parameters():
        #判断参数是否为二维或者更高维(例如权重矩阵,而不是偏置向量)
        if p.dim()>1:
            #这里采用nn.init.xavier_uniform
            nn.init.xavier_uniform_(p)
    return model.to(DEVICE)









































