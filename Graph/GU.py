import torch
import torch.nn as nn

class GRU(nn.Module):
    """
    http://zh.gluon.ai/chapter_recurrent-neural-networks/gru.html
    """
    def __init__(self, input_dim, hidden_dim):
        super(GRU, self).__init__()
        self.Wxr = nn.Linear(input_dim, hidden_dim)
        self.Whr = nn.Linear(input_dim, hidden_dim)

        self.Wxz = nn.Linear(input_dim, hidden_dim)
        self.Whz = nn.Linear(input_dim, hidden_dim)

        self.Wxh = nn.Linear(input_dim, hidden_dim)
        self.Whh = nn.Linear(input_dim, hidden_dim)

        self.acti_sigmoid = nn.Sigmoid()
        self.acti_tanh = nn.Tanh()




    def forward(self, X, H):
        # reset gate
        R = self.acti_sigmoid(self.Wxr(X) + self.Whr(H))
        # update gate
        Z = self.acti_sigmoid(self.Wxz(X)+self.Whz(H))
        # candidate hidden state
        H_ = self.acti_tanh(self.Wxh(X) + self.Whh(R * H))
        # hidden state
        output = Z * H + (1-Z) * H_

        return output


class GMULayer(nn.Module):
    """
    Gated Multimodal Layer based on 'Gated multimodal networks,
    Arevalo1 et al.' (https://arxiv.org/abs/1702.01992)
    Source Code "https://github.com/johnarevalo/gmu-mmimdb/blob/master/model.py"
    """

    def __init__(self, dim_in1, dim_in2, dim_out):
        super(GMULayer, self).__init__()
        self.dim_in1, self.dim_in2, self.dim_out = dim_in1, dim_in2, dim_out

        self.hidden1 = nn.Linear(dim_in1, dim_out, bias=False)
        self.hidden2 = nn.Linear(dim_in2, dim_out, bias=False)
        self.hidden_sigmoid = nn.Linear(dim_out * 2, dim_out, bias=False)

        # Activation functions
        self.activation = nn.Tanh()
        self.gate_activation = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.activation(self.hidden1(x1))
        h2 = self.activation(self.hidden2(x2))
        z = self.gate_activation( self.hidden_sigmoid( torch.cat((x1, x2), dim=-1) ) )

        return z * h1 + (1 - z) * h2


class SimpleGULayer(nn.Module):
    def __init__(self, d_model):
        super(SimpleGULayer, self).__init__()
        self.fc2 = nn.Linear(2 * d_model, d_model)

    def forward(self, u_i, aggr_u_i):

        concat_info = torch.cat((u_i, aggr_u_i), dim=-1)
        g_i = torch.sigmoid(self.fc2(concat_info))  # Gate factor

        # Update representation
        u_prime_i = g_i * u_i + (1 - g_i) * aggr_u_i

        return u_prime_i

def test_GMU():
    # 示例使用x
    input_dim = 10  # 输入维度
    output_dim = 10  #
    batch_size = 5
    # 创建模型
    model = GRU(input_dim, input_dim, output_dim)

    # 随机生成输入数据和隐藏状态
    x = torch.randn(batch_size, input_dim)
    h_0 = torch.randn(batch_size, input_dim)

    # 进行前向传播
    output, h_n = model(x, h_0)

    print("输出的 GRU 状态:", output)
    print("最终隐藏状态:", h_n)

def test_SimpleGU():
    model = SimpleGULayer(10)
    u = torch.randn(1, 10)
    aggr_u = torch.randn(1, 10)

    output = model(u,aggr_u)
    print(u)
    print(aggr_u)
    print(output)

if __name__ == '__main__':
    input_dim = 10  # 输入维度
    output_dim = 10  #
    batch_size = 5
    # 创建模型
    model = GMULayer(input_dim, input_dim, output_dim)

    # 随机生成输入数据和隐藏状态
    x = torch.randn(batch_size, input_dim)
    h_0 = torch.randn(batch_size, input_dim)

    # 进行前向传播
    output = model(x, h_0)

    print("输出的 GMULayer 状态:", output)
