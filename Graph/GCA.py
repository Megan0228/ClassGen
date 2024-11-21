import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from clip.model import LayerNorm, QuickGELU
from .GU import SimpleGULayer, GMULayer


class AttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        # # self.ln_1 = LayerNorm(d_model)
        # self.mlp = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(d_model, d_model * 4)),
        #     ("gelu", QuickGELU()),
        #     ("c_proj", nn.Linear(d_model * 4, d_model))
        # ]))
        # self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, need_weights=False, attn_mask: torch.Tensor = None):
        attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else attn_mask
        return self.attn(x, x, x, need_weights=need_weights, attn_mask=attn_mask)

    def save_attention_map(self, attn):
        self.attn_map = attn

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def forward(self, x: torch.Tensor, save_attn=False, attn_mask: torch.Tensor = None):
        # attn_output, attn_output_weights = self.attention(self.ln_1(x), need_weights=save_attn, attn_mask=attn_mask)
        attn_output, attn_output_weights = self.attention(x, need_weights=save_attn, attn_mask=attn_mask)

        if save_attn:
            self.save_attention_map(attn_output_weights)
            # attn_output_weights.register_hook(self.save_attn_gradients)

        # x = x + attn_output
        # x_ffn = self.mlp(self.ln_2(x))
        # x = x + x_ffn
        return attn_output

# Attention
class GCAModule(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, d_model: int, n_head: int):
        super(GCAModule, self).__init__()
        self.attn_layer1 = AttentionLayer(d_model, n_head)
        # self.attn_layer2 = AttentionLayer(d_model, n_head)
        # self.attn_layer3 = AttentionLayer(d_model, n_head)

        self.gated_layer = SimpleGULayer(d_model)
        # self.gated_layer = GMULayer(d_model, d_model, d_model)




    def forward(self, x, adj):
        '''
        x.shape: (N, in_features)
        adj: 图邻接矩阵，维度[N,N]非零即一
        '''
        # build attention mask from adj
        adj = adj.to(dtype=torch.float32, device="cpu")
        attention_mask = torch.where(adj > 0, torch.tensor(0.), torch.tensor(float('-inf')))
        attention_mask = attention_mask.to(dtype=x.dtype, device=x.device)

        x_aggr = x.unsqueeze(1)
        x_aggr = self.attn_layer1(x_aggr, attn_mask=attention_mask)
        # x_aggr = self.attn_layer2(x_aggr, attn_mask=attention_mask)
        # x_aggr = self.attn_layer3(x_aggr, attn_mask=attention_mask)
        x_aggr = x_aggr.squeeze(1)


        x_fuse = self.gated_layer(x, x_aggr)

        # x_fuse = 0.6 * x_aggr + 0.4 * x

        # # Update representation
        # x_fuse = G_factor * x + (1-G_factor) * x_aggr


        return x_fuse, x_aggr
        
if __name__ == '__main__':
    input = torch.randn(4,10) * 0.01
    # adj=torch.FloatTensor([[0,1,1,0,0,0],
    #                   [1,0,1,0,0,0],
    #                   [1,1,0,1,0,0],
    #                   [0,0,1,0,1,1],
    #                   [0,0,0,1,0,0,],
    #                   [0,0,0,1,1,0]])

    import dgl
    # 创建图并添加自环
    edges = torch.tensor([0, 1, 1, 2, 0, 2, 0, 3]), torch.tensor([1, 0, 2, 1, 2, 0, 3, 0])
    g = dgl.graph(edges)
    g = dgl.add_self_loop(g)
    adj = g.adjacency_matrix().to_dense()

    my_gat = GCAModule(10,5)
    output, x_agg = my_gat(input, adj)

    print(input)
    print(output)
    print(torch.norm(input - output, p=2, dim=-1))

    # output_norm = F.normalize(output, dim=-1)
    # print(output_norm)
    # print(torch.norm(input - output_norm, p=2, dim=-1))





        



