import torch
import torch.nn as nn
import torch.nn.functional as F
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout#dropout参数
        self.in_features = in_features#结点向量的特征维度
        self.out_features = out_features#经过GAT之后的特征维度
        self.alpha = alpha#LeakyReLU参数
        self.concat = concat# 如果为true, 再进行elu激活

        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)# xavier初始化
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)# xavier初始化

        # 定义leakyReLU激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        '''
        adj图邻接矩阵，维度[N,N]非零即一
        h.shape: (N, in_features), self.W.shape:(in_features,out_features)
        Wh.shape: (N, out_features)
        '''
        Wh = torch.mm(h, self.W) # 对应eij的计算公式
        e = self._prepare_attentional_mechanism_input(Wh)#对应LeakyReLU(eij)计算公式

        # mask
        zero_vec = -9e15*torch.ones_like(e)#将没有链接的边设置为负无穷
        attention = torch.where(adj > 0, e, zero_vec)#[N,N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留
        # 否则需要mask设置为非常小的值，因为softmax的时候这个最小值会不考虑
        attention = F.softmax(attention, dim=1)# softmax形状保持不变[N,N]，得到归一化的注意力全忠！
        attention = F.dropout(attention, self.dropout, training=self.training)# dropout,防止过拟合
        h_prime = torch.matmul(attention, Wh)#[N,N].[N,out_features]=>[N,out_features]

        # 得到由周围节点通过注意力权重进行更新后的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        # 先分别与a相乘再进行拼接
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

if __name__ == '__main__':
    input = torch.randn(6,10) * 0.01
    print(input)
    adj=torch.tensor([[0,1,1,0,0,0],
                      [1,0,1,0,0,0],
                      [1,1,0,1,0,0],
                      [0,0,1,0,1,1],
                      [0,0,0,1,0,0,],
                      [0,0,0,1,1,0]])
    my_gat = GraphAttentionLayer(10,10,0.2,0.2)
    output = my_gat(input,adj)
    print(output)
    print(input - output)
    print(torch.norm(input - output, p=2, dim=-1))
