#!/usr/bin/python

import argparse
import random
import sys

import numpy as np
import os.path as osp

from model import GenePredictionT5
from datasets import SourceDataSet, SourceDataSetLoop
from loss import MSELoss, MSELossLoop
import h5py
import params

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Trainer(object):
    """build a trainer"""
    def __init__(self, model, optimizer, criterion, data_dir, model_type):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = params.device
        self.tr_count = params.tr_count
        self.va_count = params.va_count
        self.data_dir = data_dir
        self.batch_size = params.batch_size
        self.model_type = model_type
        self.max_epoch = 1
        self.loss = 0

    def train(self):
        """training the model"""
        self.model.to(self.device)
        for epoch in range(self.max_epoch):
            # set training mode during the training process
            self.model.train()
            for count in range(self.tr_count):
                with h5py.File(self.data_dir + '/tr_{}.hdf5'.format(count + 1), 'r') as f:
                    seq_tr = np.array(f['seq'])
                    motif_tr = np.array(f['motif'])
                    fea_tr = np.array(f['fea'])
                    exp_tr = np.array(f['exp'])
                    if self.model_type == 'ScPGE':
                        tr_loader = DataLoader(SourceDataSet(seq_tr, motif_tr, fea_tr, exp_tr), 
                                       batch_size=self.batch_size, shuffle=True)
                    else:
                        loop_tr = np.array(f['loop'])
                        loop_tr[np.isinf(loop_tr)] = 0
                        tr_loader = DataLoader(SourceDataSetLoop(seq_tr, motif_tr, fea_tr, exp_tr, loop_tr), 
                                       batch_size=self.batch_size, shuffle=True)
                for _, sample_batch in enumerate(tr_loader):
                    seqs = sample_batch["seq"].float().to(self.device)
                    motif = sample_batch["motif"].float().to(self.device)
                    fea = sample_batch["fea"].float().to(self.device)
                    exp = sample_batch["exp"].float().to(self.device)
                    if self.model_type == 'ScPGE':
                        loop = None
                    else:
                        loop = sample_batch["loop"].float().to(self.device)
                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    if self.model_type == 'ScPGE-KL':
                        pred, att_weight = self.model(seqs, motif, fea)
                        loss = self.criterion(pred, exp, att_weight, loop)
                    else:
                        pred, _ = self.model(seqs, motif, fea, loop)
                        loss = self.criterion(pred, exp)
                    if np.isnan(loss.item()):
                        raise ValueError('loss is nan while training')
                    loss.backward()
                    self.optimizer.step()
            # validation and save the model with higher accuracy
            self.loss = self.validation()

        return self.loss, self.model.state_dict()

    def validation(self):
        """validate the performance of the trained model."""
        self.model.eval()
        loss_all = []
        for count in range(self.va_count):
            with h5py.File(self.data_dir + '/va_{}.hdf5'.format(count + 1), 'r') as f:
                seq_va = np.array(f['seq'])
                motif_va = np.array(f['motif'])
                fea_va = np.array(f['fea'])
                exp_va = np.array(f['exp'])
                if self.model_type == 'ScPGE':
                    va_loader = DataLoader(SourceDataSet(seq_va, motif_va, fea_va, exp_va),
                                   batch_size=self.batch_size, shuffle=False)
                else:
                    loop_va = np.array(f['loop'])
                    loop_va[np.isinf(loop_va)] = 0
                    va_loader = DataLoader(SourceDataSetLoop(seq_va, motif_va, fea_va, exp_va, loop_va),
                                   batch_size=self.batch_size, shuffle=False)
            for _, sample_batch in enumerate(va_loader):
                seqs = sample_batch["seq"].float().to(self.device)
                motif = sample_batch["motif"].float().to(self.device)
                fea = sample_batch["fea"].float().to(self.device)
                exp = sample_batch["exp"].float().to(self.device)
                if self.model_type == 'ScPGE':
                    loop = None
                else:
                    loop = sample_batch["loop"].float().to(self.device)
                if self.model_type == 'ScPGE-KL':
                    with torch.no_grad():
                        pred, att_weight = self.model(seqs, motif, fea)
                    loss = self.criterion(pred, exp, att_weight, loop)
                else:
                    with torch.no_grad():
                        pred, _ = self.model(seqs, motif, fea, loop)
                    loss = self.criterion(pred, exp)
                # 
                loss_all.append(loss.item())

        return np.mean(loss_all)


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="CNN+Transformer for predicting GEL")

    parser.add_argument("-d", dest="data_dir", type=str, default="/path/ScPGE/data/Human-DATA10-LP/K562")
    parser.add_argument("-n", dest="name", type=str, default="K562")
    parser.add_argument("-c", dest="checkpoint", type=str, default='/path/ScPGE/models/K562')
    parser.add_argument("-t", dest="type", type=str, default='ScPGE', choices=['ScPGE', 'ScPGE-LP', 'ScPGE-KL'])

    return parser.parse_args()


def main():
    args = get_args()
    random.seed(params.manual_seed)
    data_dir = args.data_dir
    name = args.name
    model_type = args.type
    fea_dim = params.fea_dim
    # implement
    loss_lowest = 10000        
    if model_type == 'ScPGE-KL':
        criterion = MSELossLoop()
    else:
        criterion = MSELoss()
    for i in range(params.trial):
        print("The program is working on the {}-th round".format(i+1))
        model = GenePredictionT5(fea_dim=fea_dim)
        # if existing multiple GPUs, and using DataParallel
        if len(params.gpu.split(',')) > 1 and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[int(id_) for id_ in params.gpu.split(',')])
        optimizer = optim.AdamW(model.parameters(), lr=params.lr_low)
        executor = Trainer(model=model,
                           optimizer=optimizer,
                           criterion=criterion,
                           data_dir=data_dir,
                           model_type=model_type)

        loss, state_dict = executor.train()
        if loss_lowest > loss:
            print("Store the weights of the model in the current run.\n")
            loss_lowest = loss
            checkpoint_file = osp.join(args.checkpoint, 'warmup.model.pth')
            torch.save({
                'model_state_dict': state_dict,
                'parameter_state_dice': optimizer.param_groups[0]
            }, checkpoint_file)


if __name__ == "__main__":
    main()

