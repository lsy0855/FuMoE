
import torch
from torch import nn
from torch.nn import Module, Embedding, Linear, Dropout, MaxPool1d, Sequential, ReLU
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F


device = "cpu" if not torch.cuda.is_available() else "cuda"

class transformer_FFN(Module):
    def __init__(self, emb_size, dropout) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.FFN = Sequential(
                Linear(self.emb_size, self.emb_size),
                ReLU(),
                Dropout(self.dropout),
                Linear(self.emb_size, self.emb_size),
                # Dropout(self.dropout),
            )
    def forward(self, in_fea):
        return self.FFN(in_fea)

def ut_mask(seq_len):
    return torch.triu(torch.ones(seq_len,seq_len),diagonal=1).to(dtype=torch.bool).to(device)

def lt_mask(seq_len):
    """ Upper Triangular Mask
    下三角掩码矩阵
    """
    return torch.tril(torch.ones(seq_len,seq_len),diagonal=-1).to(dtype=torch.bool).to(device)

def pos_encode(seq_len):
    """ position Encoding
    生成一个位置索引向量。形状为 (1, seq_len)。若 seq_len = 5，返回[[0, 1, 2, 3, 4]]
    这个向量一般用于查表式的位置嵌入（learned positional embedding）或输入到正弦位置编码函数中。
    """
    return torch.arange(seq_len).unsqueeze(0).to(device)

def get_clones(module, N):
    """ Cloning nn modules
    克隆模块函数：用于 Transformer 的层堆叠
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
```



import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import torch.nn.init as init
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch.nn.functional as F
from enum import IntEnum
from torch.nn.parameter import Parameter
import numpy as np
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, \
        MultiLabelMarginLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCELoss, MultiheadAttention
from torch.nn.functional import one_hot, cross_entropy, multilabel_margin_loss, binary_cross_entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class FuMoeCSKT(nn.Module):
    def __init__(self, n_question, n_pid, 
            d_model, n_blocks, dropout, d_ff=256, 
            loss1=0.5, loss2=0.5, loss3=0.5, start=50, num_layers=2, nheads=4, seq_len=512, r=1, gamma=1, 
            kq_same=1, final_fc_dim=512, final_fc_dim2=256, num_attn_heads=8, separate_qa=False, l2=1e-5, emb_type="qid", emb_path="", pretrain_dim=768):
        super().__init__()
      
        self.model_name = "cskt"
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        embed_l = d_model
        


        self.r = r
        self.gamma = gamma

        if self.n_pid > 0:
            if emb_type.find("scalar") != -1: 
                # print(f"question_difficulty is scalar")
                self.difficult_param = nn.Embedding(self.n_pid+1, 1) #
            else:
                self.difficult_param = nn.Embedding(self.n_pid+1, embed_l) 
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l) 
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l) 

        if emb_type.startswith("qid"):
            # n_question+1 ,d_model
            self.q_embed = nn.Embedding(self.n_question, embed_l)
            if self.separate_qa: 
                    self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
            else: # false default
                self.qa_embed = nn.Embedding(2, embed_l)
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len, 
                                    r = r, gamma=gamma)
    
        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1)
        )

        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)  # BS, seqlen,  d_model# c_ct
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            # BS, seqlen, d_model # c_ct+ g_rt =e_(ct,rt)
            qa_embed_data = self.qa_embed(target)+q_embed_data
        return q_embed_data, qa_embed_data

    def get_attn_pad_mask(self, sm):
        batch_size, l = sm.size()
        pad_attn_mask = sm.data.eq(0).unsqueeze(1)
        pad_attn_mask = pad_attn_mask.expand(batch_size, l, l)
        return pad_attn_mask.repeat(self.nhead, 1, 1)
    def forward(self, dcur, qtest=False, train=False):

        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()

        pid_data = torch.cat((q[:,0:1], qshft), dim=1).to(device)
        q_data = torch.cat((c[:,0:1], cshft), dim=1).to(device)
        target = torch.cat((r[:,0:1], rshft), dim=1).to(device)

        emb_type = self.emb_type

        # Batch First
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        if self.n_pid > 0 and emb_type.find("norasch") == -1: # have problem id

            if emb_type.find("aktrasch") == -1:
                q_embed_diff_data = self.q_embed_diff(q_data)  # 
                pid_embed_data = self.difficult_param(pid_data)  # 
                q_embed_data = q_embed_data + pid_embed_data * \
                    q_embed_diff_data  # uq *d_ct + c_ct # question encoder

            else:
                q_embed_diff_data = self.q_embed_diff(q_data)  # 
                pid_embed_data = self.difficult_param(pid_data)  # 
                q_embed_data = q_embed_data + pid_embed_data * \
                    q_embed_diff_data  # uq *d_ct + c_ct # question encoder

                qa_embed_diff_data = self.qa_embed_diff(
                    target)  # 
                qa_embed_data = qa_embed_data + pid_embed_data * \
                        (qa_embed_diff_data+q_embed_diff_data)  # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）


        y2, y3 = 0, 0
        if emb_type in ["qid", "qidaktrasch", "qid_scalar", "qid_norasch"]:
            d_output = self.model(q_embed_data, qa_embed_data)

            concat_q = torch.cat([d_output, q_embed_data], dim=-1)
            output = self.out(concat_q).squeeze(-1)
            m = nn.Sigmoid()
            preds = m(output)

        if train:
            return preds, y2, y3
        else:
            if qtest:
                return preds, concat_q
            else:
                return preds

class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, seq_len, r, gamma):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'cskt'}:
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, seq_len = seq_len, r=r, gamma=gamma)
                for _ in range(n_blocks)
            ])

    def forward(self, q_embed_data, qa_embed_data):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)


        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed
        # x = apply_monotonic_decay(x, self.gammas)
        # encoder
        
        for block in self.blocks_2:
            x = block(mask=0, query=x, key=x, values=y, apply_pos=True) #
            
        return x
    
class SoftMoEFFN(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts)  # soft gate per token

    def forward(self, x):
        B, L, D = x.size()  # (batch, seq_len, d_model)
        gate_logits = self.gate(x)                   # (B, L, num_experts)
        gate_weights = torch.softmax(gate_logits, dim=-1)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)  # (B, L, D, num_experts)
        gate_weights = gate_weights.unsqueeze(2)                                      # (B, L, 1, num_experts)
        output = (expert_outputs * gate_weights).sum(-1)                              # (B, L, D)
        return output
    
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same, seq_len, r, gamma):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same, seq_len=seq_len, r=r, gamma=gamma)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        # self.layer_norm1 = GatedRMSNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        # self.layer_norm2 = GatedRMSNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.soft_moe = SoftMoEFFN(d_model, d_ff, num_experts=8)  # 你可以设成超参
        self.dropout3 = nn.Dropout(dropout)
        self.layer_norm3 = nn.LayerNorm(d_model)
    def forward(self, mask, query, key, values, apply_pos=True):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True) #mask
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2)) # 
        query = self.layer_norm1(query) # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout( # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2)) # 
            query = self.layer_norm2(query) # lay norm
            # --- Soft-MoE FFN ---
            moe_out = self.soft_moe(query)
            query = query + self.dropout3(moe_out)
            query = self.layer_norm3(query)
        return query

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, r, gamma, seq_len=512,bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.num_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.seq_len = seq_len
        self.r = r
        self.gamma = gamma

        self.kernel_bias = ParallelKerpleLog(self.h)
        self.disentangle = DisentangledAttention(dim=self.d_model, num_heads=self.num_heads)
        self.selfattn = SelfAttention(d_model, dropout)
    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = attention(q, k, v, self.d_k,
                   mask, self.dropout, zero_pad, self.r, self.gamma, self.kernel_bias,  self.disentangle, self.selfattn,alpha=0.5)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out_proj(concat)

        return output

        
import torch
import torch.nn as nn
import torch.nn.functional as F

class DisentangledAttention(nn.Module):
    def __init__(self, dim, num_heads, max_len=512, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.pos_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # 可学习的位置偏置表（2L-1, D）
        self.pos_bias_table = nn.Parameter(torch.randn(2 * max_len - 1, dim))
        self.max_len = max_len

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: (B, H, L, D_h)
        mask: (B, L) or (B, L, L)
        """
        B, H, L, D_h = q.size()
        D = H * D_h

        # 构造相对位置嵌入索引
        pos_idx = torch.arange(L, device=q.device)
        rel_idx = pos_idx[None, :] - pos_idx[:, None] + self.max_len - 1  # (L, L)
        pos_embed = self.pos_bias_table[rel_idx]  # (L, L, D)

        # 投影到多头
        pos_embed = self.pos_proj(pos_embed)  # (L, L, D)
        pos_embed = pos_embed.view(L, L, H, D_h).permute(2, 0, 1, 3)  # (H, L, L, D_h)

        # 解耦注意力
        score_c2c = torch.matmul(q, k.transpose(-2, -1))               # (B, H, L, L)
        score_c2p = torch.einsum('bhld,hlsd->bhls', q, pos_embed)      # (B, H, L, L)
        score_p2c = torch.einsum('hlsd,bhsd->bhls', pos_embed, k)      # (B, H, L, L)

        scores = (score_c2c + score_c2p + score_p2c) / (D_h ** 0.5)

        if mask is not None:
            if mask.dim() == 2:  # (B, L)
                mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            elif mask.dim() == 3:  # (B, L, L)
                mask = mask.unsqueeze(1)  # (B, 1, L, L)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return attn
       

class SelfAttention(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.scale = d_model ** 0.5
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        B, H, L, D = q.size()
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        else:
            causal_mask = torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool))
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, H, L, L)
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        return attn



def attention(q, k, v, d_k, mask, dropout, zero_pad, r, gamma, kernel_bias,  disentangle_model, selfattn, alpha=0.5):
    scores1 = disentangle_model(q,k,v)
    scores1 = kernel_bias(scores1) # bias

    scores3 = selfattn(q,k,v,mask)
    scores3 = kernel_bias(scores3) # bias
    scores = alpha * scores1 + (1 - alpha) * scores3
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen

    # print(zero_pad)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) # set 0 in row 1
    # print(f"after zero pad scores: {scores}")
    scores = dropout(scores)
    output = torch.matmul(scores, v)

    return output

class ParallelKerpleLog(nn.Module):
    """Kernel Bias基于可学习对数核函数的并行位置偏置模块（加入时间偏置与门控机制）"""
    def __init__(self, num_attention_heads):
        super().__init__()
        self.heads = num_attention_heads
        self.num_heads_per_partition = self.heads
        self.eps = 1e-2

        def get_parameter(scale, init_method):
            if init_method == 'ones':
                return Parameter(torch.ones(self.num_heads_per_partition, dtype=torch.float32)[:, None, None] * scale)
            elif init_method == 'uniform':
                return Parameter(torch.rand(self.num_heads_per_partition, dtype=torch.float32)[:, None, None] * scale)

        self.bias_p = get_parameter(2, 'uniform')  # [H,1,1]
        self.bias_a = get_parameter(1, 'uniform')  # [H,1,1]

        self.cached_matrix = None
        self.cached_seq_len = None
        self.time_mlp = nn.Sequential(  # 时间偏置映射
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, self.heads)
        )
        self.gate = nn.Parameter(torch.zeros(self.heads))  # 每个头一个门控权重
        self.sigmoid = nn.Sigmoid()

    def stats(self):
        def get_stats(name, obj):
            return {
                name + '_mean': obj.mean().detach().cpu(),
                name + '_std': obj.std().detach().cpu(),
                name + '_max': obj.max().detach().cpu(),
                name + '_min': obj.min().detach().cpu()
            }

        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)

        dd = {}
        dd.update(get_stats('bias_a', self.bias_a))
        dd.update(get_stats('bias_p', self.bias_p))
        return dd

    def forward(self, x, delta_t=None):  # ← 新增 delta_t 可选参数
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]

        if self.cached_seq_len != seq_len_k:
            arange = torch.arange(seq_len_k, device=x.device)
            diff = (arange.view(-1, 1) - arange.view(1, -1)).abs().clamp(min=1)
            diff = diff.to(dtype=x.dtype)
            self.cached_seq_len = seq_len_k
            self.cached_matrix = diff
        else:
            diff = self.cached_matrix

        bias_p = self.bias_p.clamp(min=self.eps)  # [H,1,1]
        bias_a = self.bias_a.clamp(min=self.eps)  # [H,1,1]
        rel_bias = -bias_p * torch.log1p(bias_a * diff)  # [H, K, K]

        # === 时间偏置 ===
        if delta_t is not None:
            time_input = delta_t.unsqueeze(-1)  # [B, Q, K, 1]
            time_bias = self.time_mlp(time_input)  # [B, Q, K, H]
            time_bias = time_bias.permute(0, 3, 1, 2)  # [B, H, Q, K]
        else:
            time_bias = 0.0

        # === 门控偏置 ===
        g = self.sigmoid(self.gate).view(1, self.heads, 1, 1)
        # rel_bias: [H, K, K] → [1, H, Q, K]
        rel_bias = rel_bias.unsqueeze(0).expand(x.shape[0], -1, seq_len_q, -1)

        final_bias = g * (rel_bias + time_bias)  # [B, H, Q, K]

        return x + final_bias  # 保持输出格式不变


```


```python
import torch
import numpy as np
import os
device = "cpu" if not torch.cuda.is_available() else "cuda"

def init_model(model_name, model_config, data_config, emb_type):

    model_config = model_config.copy()
    

    model_config.pop("model_name", None)  
    model_config.pop("dataset_name", None)  
    model_config.pop("fold", None)  
    model_config.pop("learning_rate", None)  
    model_config.pop("save_dir", None)
    model_config.pop("seed", None)   
    model_config.pop("batch_size", None)
    model_config.pop("num_epochs", None)
    model_config.pop("optimizer", None)
    model_config.pop("seq_len", None)

    
    model = CSKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    return model

def load_model(model_name, model_config, data_config, emb_type, ckpt_path):
    model_config.pop("emb_type", None)  # 移除 emb_type
    model = init_model(model_name, model_config, data_config, emb_type)
    net = torch.load(ckpt_path)

    model.load_state_dict(net)
    return model
```


```python
import os, sys
import torch
import torch.nn as nn
from torch.nn.functional import one_hot, binary_cross_entropy, cross_entropy
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from torch.autograd import Variable, grad
import pandas as pd
import hashlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def cal_loss(model, ys, r, rshft, sm, preloss=[]):
    """
    model：当前训练的KT模型对象;ys：模型输出的预测结果列表（可能包含多任务输出）
    r：当前时刻的响应序列（实际答题结果）;rshft：下一时刻的响应序列（真实标签）
    sm：选择掩码（select masks），用于过滤无效交互;preloss：附加损失项列表（如正则化损失），默认为空
    """
    y = torch.masked_select(ys[0], sm) 
    t = torch.masked_select(rshft, sm) 
    y_prob = torch.sigmoid(y).clamp(1e-8, 1 - 1e-8)


    loss1 = binary_cross_entropy(y_prob.double(), t.double()) 

    if model.emb_type.find("predcurc") != -1: 
        if model.emb_type.find("his") != -1:  
            loss = model.l1*loss1+model.l2*ys[1]+model.l3*ys[2]
        else:
            loss = model.l1*loss1+model.l2*ys[1]
    elif model.emb_type.find("predhis") != -1: 
        loss = model.l1*loss1+model.l2*ys[1]
    else:
        loss = loss1
    return loss


def model_forward(model, data, rel=None):
    dcur = data

    q, c, r, t = dcur["qseqs"].to(device), dcur["cseqs"].to(device), dcur["rseqs"].to(device), dcur["tseqs"].to(device)
    qshft, cshft, rshft, tshft = dcur["shft_qseqs"].to(device), dcur["shft_cseqs"].to(device), dcur["shft_rseqs"].to(device), dcur["shft_tseqs"].to(device)

    m, sm = dcur["masks"].to(device), dcur["smasks"].to(device)

    ys, preloss = [], []

    cq = torch.cat((q[:,0:1], qshft), dim=1)
    cc = torch.cat((c[:,0:1], cshft), dim=1)
    cr = torch.cat((r[:,0:1], rshft), dim=1)

    y, y2, y3 = model(dcur, train=True)
    ys = [y[:,1:], y2, y3]
    loss = cal_loss(model, ys, r, rshft, sm, preloss)
    return loss

def get_param_hash(params_dict):

    param_str = "_".join(f"{k}={v}" for k, v in sorted(params_dict.items()))

    return hashlib.md5(param_str.encode()).hexdigest()
    
def train_model(model, train_loader, valid_loader, num_epochs, opt, save_path, ckpt_path, test_loader=None, test_window_loader=None, save_model=False, data_config=None, fold=None):

    max_auc, best_epoch = 0, -1 
    train_step = 0 
    
    rel = None
    for i in range(1, num_epochs + 1):
        loss_mean = [] 
        for data in train_loader:
            train_step+=1
            model.train()

            loss = model_forward(model, data)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_mean.append(loss.detach().cpu().numpy())
        loss_mean = np.mean(loss_mean)
        auc, acc = evaluate(model, valid_loader, model.model_name)

        if auc > max_auc+1e-3: # 性能提升超过阈值
            if save_model:

                torch.save(model.state_dict(), save_path)
            max_auc = auc
            best_epoch = i
            testauc, testacc = -1, -1
            window_testauc, window_testacc = -1, -1
            if not save_model:
                if test_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_predictions.txt")
                    testauc, testacc = evaluate(model, test_loader, model.model_name, save_test_path)
                if test_window_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_window_predictions.txt")
                    window_testauc, window_testacc = evaluate(model, test_window_loader, model.model_name, save_test_path)
            validauc, validacc = auc, acc

        if i % 20 == 0 or i == num_epochs or i - best_epoch >= 10:
            print(f"Epoch: {i}, validauc: {validauc:.4f}, validacc: {validacc:.4f}, best epoch: {best_epoch}, best auc: {max_auc:.4f}, train loss: {loss_mean}, emb_type: {model.emb_type}")
            print(f"            testauc: {round(testauc, 4)}, testacc: {round(testacc, 4)}, window_testauc: {round(window_testauc, 4)}, window_testacc: {round(window_testacc, 4)}")
            last_epoch_logged = i


        if i - best_epoch >= 10:
            if last_epoch_logged != i:
                print(f"[Early Stop] Epoch: {i}, validauc: {validauc:.4f}, validacc: {validacc:.4f}, best epoch: {best_epoch}, best auc: {max_auc:.4f}, train loss: {loss_mean}, emb_type: {model.emb_type}")
                print(f"            testauc: {round(testauc, 4)}, testacc: {round(testacc, 4)}, window_testauc: {round(window_testauc, 4)}, window_testacc: {round(window_testacc, 4)}")
            break

    return testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch
```


```python
import os, sys
import torch
from torch.utils.data import DataLoader
import numpy as np

def set_seed(seed):
    """Set the global random seed.
    
    Args:
        seed (int): random seed
    """
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        print("Set seed failed,details are ", e)
        pass
    import numpy as np
    np.random.seed(seed)
    import random as python_random
    python_random.seed(seed)
    # cuda env
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import datetime
def get_now_time():

    now = datetime.datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    return dt_string

def debug_print(text,fuc_name=""):

    print(f"{get_now_time()} - {fuc_name} - said: {text}")

```


```python
import numpy as np
import torch
from torch import nn
from torch.nn.functional import one_hot
from sklearn import metrics
import pandas as pd
import csv

def evaluate(model, test_loader, model_name, rel=None, save_path=""):

    with torch.no_grad(): 
        y_trues = [] 
        y_scores = [] 
        dres = dict() 
        test_mini_index = 0 
        for data in test_loader:
           
            dcur = data
            q, c, r = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"] 

            qshft, cshft, rshft= dcur["shft_qseqs"], dcur["shft_cseqs"], dcur["shft_rseqs"]
            m, sm = dcur["masks"], dcur["smasks"]
            q, c, r, qshft, cshft, rshft, m, sm = q.to(device), c.to(device), r.to(device), qshft.to(device), cshft.to(device), rshft.to(device), m.to(device), sm.to(device)
            model.eval()
 
            cq = torch.cat((q[:,0:1], qshft), dim=1) # 题目历史序列
            cc = torch.cat((c[:,0:1], cshft), dim=1) # 知识点历史序列
            cr = torch.cat((r[:,0:1], rshft), dim=1) # 响应历史序列

            y = model(dcur)
            y = y[:,1:]

            if save_path != "":
                result = save_cur_predict_result(dres, c, r, cshft, rshft, m, sm, y)
                fout.write(result+"\n")

            y = torch.masked_select(y, sm).detach().cpu() 
   
            t = torch.masked_select(rshft, sm).detach().cpu() 

            y_trues.append(t.numpy())
            y_scores.append(y.numpy())
            test_mini_index+=1

        ts = np.concatenate(y_trues, axis=0)
        ps = np.concatenate(y_scores, axis=0) 

        auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
        prelabels = [1 if p >= 0.5 else 0 for p in ps] 
        acc = metrics.accuracy_score(ts, prelabels)
    return auc, acc
```


```python
import torch
from torch import nn
from torch.nn import Module, Embedding, Linear, Dropout, MaxPool1d, Sequential, ReLU
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F



#!/usr/bin/env python
# coding=utf-8

import os, sys
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor
else:
    from torch import FloatTensor, LongTensor

class KTDataset(Dataset): # 继承PyTorch的Dataset类
    """ 自定义数据集类，支持针对 KT 模型（如 CSKT、SAKT、DKVMN）构建训练/验证/测试数据，支持概念评估和问题评估。
        file_path：原始数据文件路径。
        input_type：指定输入类型（题目ID或知识点ID）。
        folds：交叉验证的折数集合（如{0,1}）。
        qtest：是否为问题级别评估模式。
    """

    def __init__(self, file_path, input_type, folds, qtest=False):
        super(KTDataset, self).__init__()
        sequence_path = file_path 
        self.input_type = input_type 
        self.qtest = qtest 
 
        folds = sorted(list(folds)) 
        folds_str = "_" + "_".join([str(_) for _ in folds])
        if self.qtest: 
            processed_data = file_path + folds_str + "_qtest.pkl"
        else:
            processed_data = os.path.join("../autodl-tmp/kaggle/working", f"{os.path.basename(file_path)}_processed_{input_type}.pkl")

  
        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            if self.qtest:      
                self.dori, self.dqtest = self.__load_data__(sequence_path, folds) 
                save_data = [self.dori, self.dqtest]
            else: 
                self.dori = self.__load_data__(sequence_path, folds) 
                save_data = self.dori
            pd.to_pickle(save_data, processed_data) 
        else: 
            print(f"Read data from processed file: {processed_data}")
            if self.qtest:
                self.dori, self.dqtest = pd.read_pickle(processed_data)
            else:
                self.dori = pd.read_pickle(processed_data)
                for key in self.dori:
                    self.dori[key] = self.dori[key]#[:100]
        print(f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}")

    def __len__(self):

        return len(self.dori["rseqs"])

    def __getitem__(self, index): 

        dcur = dict()
 
        mseqs = self.dori["masks"][index] 
        for key in self.dori: 
            if key in ["masks", "smasks"]:
                continue
            if len(self.dori[key]) == 0: 
                dcur[key] = self.dori[key]
                dcur["shft_"+key] = self.dori[key]
                continue

            seqs = self.dori[key][index][:-1] * mseqs 
  
            shft_seqs = self.dori[key][index][1:] * mseqs

            dcur[key] = seqs 
            dcur["shft_"+key] = shft_seqs 
        dcur["masks"] = mseqs 
        dcur["smasks"] = self.dori["smasks"][index]

        if not self.qtest:
            return dcur
        else:
            dqtest = dict()
            for key in self.dqtest:
                dqtest[key] = self.dqtest[key][index]
            return dcur, dqtest

    def __load_data__(self, sequence_path, folds, pad_val=-1):

        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": []}


        df = pd.read_csv(sequence_path)
        df = df[df["fold"].isin(folds)] 
        interaction_num = 0

        dqtest = {"qidxs": [], "rests":[], "orirow":[]}
        for i, row in df.iterrows(): 


            if "concepts" in self.input_type: 
                try:
                    dori["cseqs"].append([int(_) for _ in row["concepts"].split(",")])
                except ValueError:

                    continue
                

            if "questions" in self.input_type:
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])

            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])

            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])

            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])

            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            interaction_num += dori["smasks"][-1].count(1)

            if self.qtest:
                dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
                dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])

        for key in dori:
            try:
                if key not in ["rseqs"]:
                    dori[key] = LongTensor(dori[key])

                else:
                    dori[key] = FloatTensor(dori[key])
  
            except Exception as e:
                print(f"Error processing key {key}: {e}")
                raise


        mask_seqs = (dori["cseqs"][:,:-1] != pad_val) * (dori["cseqs"][:,1:] != pad_val)
        dori["masks"] = mask_seqs

        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        print(f"interaction_num: {interaction_num}")


        if self.qtest:
            for key in dqtest:
                dqtest[key] = LongTensor(dqtest[key])[:, 1:]
            
            return dori, dqtest

        return dori
```


```python
import os, sys
import json

from torch.utils.data import DataLoader
import numpy as np

def init_dataset4train(dataset_name, model_name, data_config, i, batch_size, diff_level=None, args=None, not_select_dataset=None, re_mapping=False):

    print(f"dataset_name:{dataset_name}")
    print(f"data_config:{data_config}")
    data_config = data_config[dataset_name] 
    all_folds = set(data_config["folds"]) 
  

    curvalid = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], {i})
    curtrain = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], all_folds - {i})

    train_loader = DataLoader(curtrain, batch_size=batch_size)
    valid_loader = DataLoader(curvalid, batch_size=batch_size)
    
 

    return train_loader, valid_loader#, test_loader, test_window_loader
```


```python
import os
import argparse
import json

import torch
torch.set_num_threads(4) 
from torch.optim import SGD, Adam
import copy 

import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 
device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'

def main(params):
    set_seed(params["seed"])
    model_name, dataset_name, fold, emb_type, save_dir = params["model_name"], params["dataset_name"], \
        params["fold"], params["emb_type"], params["save_dir"]
        
    debug_print(text = "load config files.",fuc_name="main")

    model_config = copy.deepcopy(params) 
    model_config.pop("emb_type", None) 
    batch_size, num_epochs, optimizer = params["batch_size"], params["num_epochs"], params["optimizer"]
    data_config = {
        "statics2011": {
        "dpath": "../autodl-tmp/kaggle/input/al_2005_train/maxlen20", 
        "num_q": 0, 
        "num_c": 1223, 
        "input_type": [
            "concepts"
        ],
        "max_concepts": 1,
 
        "min_seq_len": 3, 
        "maxlen": 20,  

        "emb_path": "", 
        "train_valid_original_file": "train_valid.csv", 
        "train_valid_file": "train_valid_sequences.csv", 
        "folds": [
            0,
            1,
            2,
            3,
            4
        ],
        "test_original_file": "test.csv",  
        "test_file": "test_sequences.csv", 
        "test_window_file": "test_window_sequences.csv",
        "train_valid_original_file_quelevel": "train_valid_quelevel.csv",
        "train_valid_file_quelevel": "train_valid_sequences_quelevel.csv",
        "test_file_quelevel": "test_sequences_quelevel.csv",
        "test_window_file_quelevel": "test_window_sequences_quelevel.csv",
        "test_original_file_quelevel": "test_quelevel.csv"
    }
    }
    print("Start init data")
    print(dataset_name, model_name, data_config, fold, batch_size)
    
    debug_print(text="init_dataset",fuc_name="main")

    train_loader, valid_loader, *_ = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size)

    params_str = "_".join([str(v) for k,v in params.items() if not k in ['other_config']])
    ckpt_path = "../autodl-tmp/kaggle/working/"
    print("start training model")
    learning_rate = params["learning_rate"]
        
    debug_print(text = "init_model",fuc_name="main")

    model = init_model(model_name,model_config, data_config[dataset_name], emb_type)
    if optimizer == "sgd":
        # opt = SGD(model.parameters(), learning_rate, momentum=0.9)
        opt = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        # 1e-4, 5e-4, 最多到 1e-3
        opt = Adam(model.parameters(), learning_rate, weight_decay=1e-3, amsgrad=True)
    testauc, testacc = -1, -1
    window_testauc, window_testacc = -1, -1
    validauc, validacc = -1, -1
    best_epoch = -1
    save_model = True
    
    debug_print(text = "train model",fuc_name="main")

    param_hash = get_param_hash(params) 

    save_subdir = os.path.join(params["save_dir"], param_hash)
    os.makedirs(save_subdir, exist_ok=True)
    params["save_dir"] = os.path.join(save_subdir, f"{params['emb_type']}_model.ckpt")

    testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch =  train_model(model, train_loader, valid_loader, num_epochs, opt,params["save_dir"], ckpt_path, None, None, save_model)
    
    if save_model:

        best_model = init_model(model_name, model_config, data_config[dataset_name], emb_type)
        model_path = params["save_dir"]
        if os.path.exists(model_path):
            net = torch.load(model_path)
            model.load_state_dict(net)

        best_model.load_state_dict(net)

    print("fold\tmodelname\tembtype\ttestauc\ttestacc\twindow_testauc\twindow_testacc\tvalidauc\tvalidacc\tbest_epoch")
    print(str(fold) + "\t" + model_name + "\t" + emb_type + "\t" + str(round(testauc, 4)) + "\t" + str(round(testacc, 4)) + "\t" + str(round(window_testauc, 4)) + "\t" + str(round(window_testacc, 4)) + "\t" + str(validauc) + "\t" + str(validacc) + "\t" + str(best_epoch))

    print(f"end:{datetime.datetime.now()}")
    score_auc = validauc
    score_acc = validacc
    if score_auc is None:
        score_auc = 0  
    if score_acc is None:
        score_acc = 0
    return score_auc,score_acc
```


```python
import itertools
import random

param_grid = {
    "gamma": [1],
    "r": [1],
    "learning_rate": [1e-3],
    "d_ff": [128],
    "n_blocks": [2],
    "num_attn_heads": [8],
    "dropout": [0.5],
    "batch_size": [256],
    "fold": [2],
    "d_model": [128],
    "final_fc_dim": [128]
}

# 固定参数
fixed_params = {
    "model_name": "cskt",
    "emb_type": "qid",
    "dataset_name": "statics2011",

    "final_fc_dim2": 128,

    "save_dir": "saved_model",
    "seed": 3407,
    "num_epochs": 200,
    "optimizer": "adam",
}


keys = list(param_grid.keys())
all_combinations = list(itertools.product(*(param_grid[k] for k in keys)))# 生成所有参数组合


n_samples = 1
random.seed(42)
sampled_combinations = random.sample(all_combinations, n_samples)



best_score = -1
best_params = None
best_model_path = None

for i, combo in enumerate(sampled_combinations):
    trial_params = dict(zip(keys, combo))
    params = {**fixed_params, **trial_params}
    
    print(f"[{i+1}/{n_samples}] Training with params: {params}")
    
    score_auc, score_acc = main(params) # 返回评估指标（如 valid AUC）

    print(f"Score: {score_auc}")

    if score_auc is not None and score_auc > best_score:
        # 删除上一轮保存的模型（如果有）
        if best_model_path is not None and os.path.exists(best_model_path):
            os.remove(best_model_path)
        best_score = score_auc
        best_params = params
        best_model_path = params["save_dir"]
    else:
        # 当前模型不是最优，直接删除
        if os.path.exists(params["save_dir"]):
            os.remove(params["save_dir"])
print(f"Best model saved at: {best_model_path}")
print("\n Best score:", best_score)
print("Best params:", best_params)


import os, sys
import json

from torch.utils.data import DataLoader
import numpy as np

def init_test_datasets(data_config, model_name, batch_size, diff_level=None, args=None, re_mapping=False):
    test_question_loader, test_question_window_loader = None, None
    
    test_dataset = KTDataset(os.path.join(data_config["dpath"], 
                                          data_config["test_file"]), 
                             data_config["input_type"], 
                             {-1})

    test_window_dataset = KTDataset(os.path.join(data_config["dpath"], 
                                                 data_config["test_window_file"]), 
                                    data_config["input_type"], {-1})
    if "test_question_file" in data_config:
        test_question_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config["input_type"], {-1}, True)
        test_question_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"], {-1}, True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_window_loader = DataLoader(test_window_dataset, batch_size=batch_size, shuffle=False)
    if "test_question_file" in data_config:
        print(f"has test_question_file!")
        test_question_loader,test_question_window_loader = None,None
        if not test_question_dataset is None:
            test_question_loader = DataLoader(test_question_dataset, batch_size=batch_size, shuffle=False)
        if not test_question_window_dataset is None:
            test_question_window_loader = DataLoader(test_question_window_dataset, batch_size=batch_size, shuffle=False)

    return test_loader, test_window_loader, test_question_loader, test_question_window_loader
```


```python
import os
import argparse
import json
import copy
import torch
import pandas as pd

device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'


def main1(params1):
    data_config1 = {
        "dpath": "../autodl-tmp/kaggle/input/al_2005_train/maxlen50", 
        "num_q": 0, 
        "num_c": 1223, 
        "input_type": [
            "concepts"
        ],
        "max_concepts": 1,
。
        "min_seq_len": 3, 
        "emb_path": "", 
        "train_valid_original_file": "train_valid.csv",
        "train_valid_file": "train_valid_sequences.csv", 
        "folds": [
            0,
            1,
            2,
            3,
            4
        ],
        "test_original_file": "test.csv",  
        "test_file": "test_sequences.csv", 
        "test_window_file": "test_window_sequences.csv",
        "train_valid_original_file_quelevel": "train_valid_quelevel.csv",
        "train_valid_file_quelevel": "train_valid_sequences_quelevel.csv",
        "test_file_quelevel": "test_sequences_quelevel.csv",
        "test_window_file_quelevel": "test_window_sequences_quelevel.csv",
        "test_original_file_quelevel": "test_quelevel.csv"
    
    }


    save_dir, batch_size = params1["save_dir"], params1["batch_size"]

    model_name = 'cskt'
    test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config1, model_name, batch_size)

    print(f"Start predicting model: {model_name}, save_dir: {save_dir}")

    print(f"data_config1: {data_config1}")
    model_config = copy.deepcopy(params1) 

    emb_type = "qid"
    model = load_model(model_name, model_config, data_config1, emb_type, save_dir)
   
    save_test_path = os.path.join(save_dir, model.emb_type+"_test_predictions.txt")

    testauc, testacc = evaluate(model, test_loader, model_name, save_test_path)
    print(f"testauc: {testauc}, testacc: {testacc}")

    window_testauc, window_testacc = -1, -1
    save_test_window_path = os.path.join(save_dir, model.emb_type+"_test_window_predictions.txt")
    if model.model_name == "rkt":
        window_testauc, window_testacc = evaluate(model, test_window_loader, model_name, rel, save_test_window_path)
    else:
        window_testauc, window_testacc = evaluate(model, test_window_loader, model_name, save_test_window_path)
    print(f"testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}")

    # 初始化结果字典
    dres = {
        "testauc": testauc, "testacc": testacc, "window_testauc": window_testauc, "window_testacc": window_testacc,
    }  

    q_testaucs, q_testaccs = -1,-1
    qw_testaucs, qw_testaccs = -1,-1
    if "test_question_file" in data_config1 and not test_question_loader is None:
        save_test_question_path = os.path.join(save_dir, model.emb_type+"_test_question_predictions.txt")
        q_testaucs, q_testaccs = evaluate_question(model, test_question_loader, model_name, fusion_type, save_test_question_path)
        for key in q_testaucs:
            dres["oriauc"+key] = q_testaucs[key]
        for key in q_testaccs:
            dres["oriacc"+key] = q_testaccs[key]
            
    if "test_question_window_file" in data_config1 and not test_question_window_loader is None:
        save_test_question_window_path = os.path.join(save_dir, model.emb_type+"_test_question_window_predictions.txt")
        qw_testaucs, qw_testaccs = evaluate_question(model, test_question_window_loader, model_name, fusion_type, save_test_question_window_path)
        for key in qw_testaucs:
            dres["windowauc"+key] = qw_testaucs[key]
        for key in qw_testaccs:
            dres["windowacc"+key] = qw_testaccs[key]   
    print(dres)
```


```python
# del params['emb_type']
# del best_params['fold']
params1 = best_params
# 打印参数配置
print("运行参数配置:", params1)
# 调用主函数
main1(params1)

    
