#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.criteria = nn.MSELoss()

    def forward(self, prediction, target):
        prediction = prediction.view(-1)
        target = target.view(-1)
        loss = self.criteria(prediction, target)
        return loss
        

class MSELossLoop(nn.Module):
    def __init__(self):
        super(MSELossLoop, self).__init__()
        self.criteria1 = nn.MSELoss()
        self.criteria2 = nn.KLDivLoss()
        self.alpha = 1

    def normalization(self, x):
        _, M = x.size()
        M = M // 2
        x_part = torch.concat((x[:, :M], x[:, (M+1):]), dim=1)
        x_max, _ = torch.max(x_part, dim=1, keepdim=True)
        x_min, _ = torch.min(x_part, dim=1, keepdim=True)
        x_norm = (x - x_min) / (x_max - x_min)
        x_norm[:, M] = 0
        return x_norm

    def decay(self, M, N):
        x = torch.arange(M)
        x -= M // 2
        # val = x**2 / 2
        # y = torch.exp(-val) / torch.sqrt(2 * torch.pi)
        val = torch.abs(x) / 2
        y = torch.exp(-val)
        return y.unsqueeze(0).repeat(N, 1)

    def forward(self, pred_gene, target_gene, attn_weight, loop):
        # target_gene shape: N*1
        # loop shape: N*M*M
        attn_integrated = torch.zeros_like(attn_weight[0])
        for att in attn_weight:
            attn_integrated += att
        attn_integrated = torch.mean(attn_integrated, dim=1)
        N, M, _ = attn_integrated.size()
        att_gene = attn_integrated[:, M//2, :] # N*M
        att_gene = self.normalization(att_gene)
        #
        loop_gene = loop[:, M//2, :]
        dis_decay = self.decay(M, N)
        loop_gene += dis_decay.to(att_gene.device)
            
        loop_gene = self.normalization(loop_gene)
        loss1 = self.criteria1(pred_gene, target_gene)
        loss2 = self.criteria2(att_gene, loop_gene)
        loss = loss1 + self.alpha * loss2

        return loss