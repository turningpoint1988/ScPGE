# -*- coding: utf8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


# Relative Position Encoding ##
class RelativePositionBias(nn.Module):
    def __init__(self, num_buckets=32, max_distance=128, n_heads=4):
        super(RelativePositionBias, self).__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets, max_distance):
        num_buckets //= 2
        ret = (relative_position < 0).to(relative_position) * num_buckets
        relative_position = torch.abs(relative_position)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(relative_position / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1), )

        ret += torch.where(is_small, relative_position.long(), val_if_large)
        return ret.long()

    def forward(self, relative_position):
        rp_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        rp_bias = self.relative_attention_bias(rp_bucket)
        return rp_bias

# End ##


# Ingredients of Transformer ##
class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, bias, loop=None):
        attn = torch.matmul(q / self.scale, k.transpose(-1, -2))
        if bias is not None:
            attn += bias
        if loop is not None:
            attn *= loop
        attn = F.softmax(attn, dim=-1)
        attn_m = self.dropout(attn)
        output = torch.matmul(attn_m, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        if d_model % n_head != 0:
            raise ValueError("The hidden size is not a multiple of the number of attention heads")

        self.n_head = n_head
        self.d_k = d_model // n_head
        self.fc_query = nn.Linear(d_model, d_model, bias=False)
        self.fc_key = nn.Linear(d_model, d_model, bias=False)
        self.fc_value = nn.Linear(d_model, d_model, bias=False)
        self.attention = ScaledDotProductAttention(scale=self.d_k ** 0.5, dropout=dropout)
        self.fc_out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        """
        x has shape (*, L, C)
        return shape (*, nhead, L, C/nhead)
        """
        new_shape = x.shape[:-1] + (self.n_head, -1)
        x = x.view(*new_shape)
        return x.transpose(-3, -2)

    def forward(self, x, bias, loop=None):
        q = self.transpose_for_scores(self.fc_query(x))
        k = self.transpose_for_scores(self.fc_key(x))
        v = self.transpose_for_scores(self.fc_value(x))
        x, attn = self.attention(q, k, v, bias, loop)
        x = x.transpose(-3, -2)
        x = x.reshape(*x.shape[:-2], -1)
        x = self.dropout(self.fc_out(x))
        return x, attn


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ff(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model=d_model, n_head=n_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout)

    def forward(self, x, bias, loop=None):
        branch, attn = self.attn(self.norm1(x), bias, loop)
        x = x + branch
        x = x + self.ffn(self.norm2(x))
        return x, attn


class TransformerEncoder(nn.Module):
    def __init__(self, n_layer, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(**kwargs) for _ in range(n_layer)])

    def forward(self, x, bias, loop=None):
        attn_weight = []
        for module in self.layers:
            x, w = module(x, bias, loop)
            attn_weight.append(w)
        return x, attn_weight
# End ##


class Rconv_block(nn.Module):
    def __init__(self, out_channel, kernel_size=1, bias=False):
        """Residual block"""
        super(Rconv_block, self).__init__()
        padding = kernel_size // 2
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.GELU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=padding, bias=bias)
        )

    def forward(self, x):
        x = x + self.conv_block(x)
        return x


class GenePredictionT5(nn.Module):
    def __init__(self, motiflen=30, fea_dim=4):
        """Input size is N*4*M*L, M means the number of cCREs and L means the sequence length."""
        super(GenePredictionT5, self).__init__()
        out_channel = 128
        tower_num = 3 
        part = 3 
        self.stem_seq = nn.Sequential(
            nn.Conv2d(4, out_channel, kernel_size=(1, motiflen)), # new
            Rconv_block(out_channel, kernel_size=1, bias=False),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)), 
            nn.Dropout(p=0.2)
        )
        self.stem_motif = nn.Sequential(
            nn.Conv2d(1, out_channel, kernel_size=(1, motiflen)), # new
            Rconv_block(out_channel, kernel_size=1, bias=False),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)), 
            nn.Dropout(p=0.2)
        )
        self.stem_fea = nn.Sequential(
            nn.Conv2d(fea_dim, out_channel, kernel_size=(1, motiflen)), # new
            Rconv_block(out_channel, kernel_size=1, bias=False),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)), 
            nn.Dropout(p=0.2)
        )
        self.conv_tower = nn.ModuleList([nn.Sequential(
            nn.BatchNorm2d(out_channel*part),
            nn.GELU(),
            nn.Conv2d(out_channel*part, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            Rconv_block(out_channel, kernel_size=1, bias=False),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)), 
            nn.Dropout(p=0.2)
        )] + [nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.GELU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            Rconv_block(out_channel, kernel_size=1, bias=False),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(p=0.2)
        ) for _ in range(tower_num)])
        # Notice: before Transformer, we need to reshape outputs as N*M*(4*L)
        in_channel = 8 * out_channel
        d_model = 256
        n_layer = 4 
        n_head = 4
        self.mask = True
        self.projection = nn.Sequential(
            nn.Linear(in_channel, d_model),
            nn.Dropout(p=0.2)
        )
        # relative position
        self.rp_bias = RelativePositionBias(
            num_buckets=64, max_distance=256, n_heads=n_head
        )
        self.transformer = TransformerEncoder(
            n_layer=n_layer,
            d_model=d_model,
            n_head=n_head,
            dim_feedforward=d_model * 4,
            dropout=0.1,
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
                    nn.Linear(d_model, 64),
                    nn.GELU(),
                    nn.Dropout(p=0.1),
                    nn.Linear(64, 1)
                    )
        self.activation = nn.Softplus()
        print(self.modules())

    def _relative_position(self, pos_rp):
        rp_matrix = pos_rp.unsqueeze(-1) - pos_rp.unsqueeze(-2)
        rp_bias = self.rp_bias(rp_matrix)
        rp_bias = rp_bias.permute(2, 0, 1)

        return rp_bias

    def forward(self, seq, motif, fea, loop=None):
        """
        :param seq: N*4*M*L, M means the number of cCREs and L means the sequence length
        :param seq_rc: N*4*M*L
        :param fea: N*4*M*L
        :param loop: M*M
        :return: N*1
        """
        N, _, M, L = seq.size()
        x_seq = self.stem_seq(seq)
        x_motif = self.stem_motif(motif)
        x_fea = self.stem_fea(fea)
        x = torch.cat((x_seq, x_motif, x_fea), dim=1)
        for module in self.conv_tower:
            x = module(x)
        # reshape x into N*M*128*(L//64)
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(N, M, -1)
        x = self.projection(x)
        if self.mask:
            attn_mask = (torch.eye(M, device=seq.device)>0)
            msk_bias = torch.zeros((M, M), device=seq.device)
            msk_bias = msk_bias.masked_fill(attn_mask, float('-inf'))
        else:
            msk_bias = torch.zeros((M, M), device=seq.device)
        if loop is not None:
            loop = torch.unsqueeze(loop, dim=1)
            loop = torch.exp(loop)
        # relative position
        pos_rp = torch.arange(M, device=seq.device)
        rp_bias = self._relative_position(pos_rp)
        x, attn_weight = self.transformer(x, rp_bias+msk_bias, loop)

        x = self.layer_norm(x)
        x = x[:, M // 2, :]
        x = self.fc(x)
        x = self.activation(x)

        return x, attn_weight