"""Params"""
import torch
import os.path as osp

# params for dataset and data loader
species = 'Human' # Mouse
batch_size = 64 
fea_dim = 4

# params for training network
gpu = '0,1'
trial = 12
lr = 1e-05 
lr_up = 1e-03
lr_low = 1e-06 
pre_step = 5000 
train_step = 8000 
eval_step = 1000 
num_epochs = 50
manual_seed = 666
if torch.cuda.is_available():
    torch.cuda.manual_seed(manual_seed)
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    torch.manual_seed(manual_seed)
gradient_clip = 0.2 # 0.2

if species == 'Human':
    tr_count = 9
    te_count = 1
    va_count = 1
else:
    tr_count = 10
    te_count = 2
    va_count = 1

