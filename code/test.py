#!/usr/bin/python

import os
import sys
import argparse
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# custom functions defined by user
from model import GenePredictionT5
from datasets import SourceDataSet, SourceDataSetLoop
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import h5py
import params
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def scatterplot(pred, true, out_f):
    df = pd.DataFrame({'Prediction': pred, 'True': true})
    max_p, max_t = int(np.ceil(np.max(pred))), int(np.ceil(np.max(true)))
    x = range(0, np.maximum(max_p, max_t))
    sns.set_theme(style="dark")
    fig, ax = plt.subplots()
    sns.despine(fig)
    sns.scatterplot(data=df, x="Prediction", y="True", s=10, ax=ax)
    ax.plot(x, x, '--', linewidth=0.8, color='grey')
    plt.savefig(out_f, format='png', bbox_inches='tight', dpi=300)


def test(model, data_dir, model_type):
    # set eval state for Dropout and BN layers
    model.eval()
    p_all = []
    t_all = []
    for count in range(params.te_count):
        # load data
        with h5py.File(data_dir + '/te_{}.hdf5'.format(count + 1), 'r') as f:
            seq_te = np.array(f['seq'])
            motif_te = np.array(f['motif'])
            fea_te = np.array(f['fea'])
            exp_te = np.array(f['exp'])
            if model_type == 'ScPGE':
                te_loader = DataLoader(SourceDataSet(seq_te, motif_te, fea_te, exp_te),
                               batch_size=params.batch_size, shuffle=False)
            else:
                loop_te = np.array(f['loop'])
                loop_te[np.isinf(loop_te)] = 0
                te_loader = DataLoader(SourceDataSetLoop(seq_te, motif_te, fea_te, exp_te, loop_te),
                               batch_size=params.batch_size, shuffle=False)
        for step, sample_batch in enumerate(te_loader):
            seqs = sample_batch["seq"].float().to(params.device)
            motif = sample_batch["motif"].float().to(params.device)
            fea = sample_batch["fea"].float().to(params.device)
            exp = sample_batch["exp"].float()
            if model_type == 'ScPGE':
                loop = None
            else:
                loop = sample_batch["loop"].float().to(params.device)
            if model_type == 'ScPGE-KL':
                with torch.no_grad():
                    pred, _ = model(seqs, motif, fea)
            else:
                with torch.no_grad():
                    pred, _ = model(seqs, motif, fea, loop)
            pred = pred.view(-1).data.cpu().numpy()
            exp = exp.view(-1).data.cpu().numpy()
            if count == 0 and step == 0:
                p_all = pred
                t_all = exp
            else:
                p_all = np.concatenate((p_all, pred))
                t_all = np.concatenate((t_all, exp))

    pr = pearsonr(t_all, p_all)[0]
    mae = mean_absolute_error(t_all, p_all)
    print("pearson: {:.3f}\tmae: {:.3f}\n".format(pr, mae))

    return pr, mae, p_all, t_all


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="CNN+Transformer for predicting GEL")

    parser.add_argument("-d", dest="data_dir", type=str, default="/path/ScPGE/data/Human-DATA10/K562")
    parser.add_argument("-n", dest="name", type=str, default="K562")
    parser.add_argument("-c", dest="checkpoint", type=str, default='/path/ScPGE/models/K562')
    parser.add_argument("-t", dest="type", type=str, default='ScPGE', choices=['ScPGE', 'ScPGE-LP', 'ScPGE-KL'])

    return parser.parse_args()


def main():
    """Create the model and start the training."""
    args = get_args()
    data_dir = args.data_dir
    name = args.name
    model_type = args.type
    fea_dim = params.fea_dim
    f_out = open(osp.join(args.checkpoint, 'score.txt'), 'a')
    # Load weights
    checkpoint_file = osp.join(args.checkpoint, 'model.best.pth')
    chk = torch.load(checkpoint_file, map_location='cuda:0')
    state_dict = chk['model_state_dict']
    model = GenePredictionT5(fea_dim=fea_dim)
    # if existing multiple GPUs, and using DataParallel
    if len(params.gpu.split(',')) > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[int(id_) for id_ in params.gpu.split(',')])
    model.load_state_dict(state_dict)
    model.to(params.device)

    pr, mae, p_all, t_all = test(model, data_dir, model_type)
    f_out.write("{}\tpr: {:.3f}\tmae: {:.3f}\n".format(name, pr, mae))
    f_out.close()
    # plot 
    out_f = osp.join(args.checkpoint, 'pr.png')
    scatterplot(p_all, t_all, out_f)


if __name__ == "__main__":
    main()

