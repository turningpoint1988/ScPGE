#!/usr/bin/python
import os
import sys
import argparse
import numpy as np
import os.path as osp
from Bio import SeqIO
import pyBigWig
import json
import re

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

# custom functions defined by user
from model import GenePredictionT5
from datasets import SourceDataSet, SourceDataSetLoop
import h5py
import pandas as pd
from scipy.signal import convolve2d


SEQ_LEN = 600
WINDOW = 600
NUMBER = 100
THRESHOLDS = [10, 50, 100]
INDEX = ['chr' + str(i + 1) for i in range(23)]
INDEX[22] = 'chrX'
seq_rc_dict = {'A': 'T', 'G': 'C',
               'C': 'G', 'T': 'A',
               'a': 'T', 'g': 'C',
               'c': 'G', 't': 'A'}


def MGPUtoSingle(state_dict):
    from collections import OrderedDict
    
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:]  # delete `module.`
        name = k.replace("module.", "")
        state_dict_new[name] = v
        
    return state_dict_new


def gene_selection(gene_file, cCRE_file):
    cCRE_set = {}
    with open(cCRE_file) as f:
        for line in f:
            line_split = line.strip().split('\t')
            chrom = line_split[0]
            start = int(line_split[1])
            end = int(line_split[2])
            mid = (start + end) // 2
            if chrom not in cCRE_set.keys():
                cCRE_set[chrom] = [mid]
            else:
                cCRE_set[chrom] += [mid]
    # merge adjacent regions
    for chrom in cCRE_set.keys():
        cCRE_pos = cCRE_set[chrom]
        cCRE_pos.sort()
        print("The number of cCREs on {} is {}".format(chrom, len(cCRE_set[chrom])))
        cCRE_pos_new = []
        start = 0
        end = start + 1
        while start < len(cCRE_pos):
            pos_start = cCRE_pos[start]
            while end < len(cCRE_pos):
                pos_end = cCRE_pos[end]
                if pos_end - pos_start > WINDOW:
                    tmp = cCRE_pos[start:end]
                    tmp = tmp[len(tmp)//2]
                    cCRE_pos_new.append(tmp)
                    start = end
                    end = start + 1
                    break
                else:
                    end += 1
            if end >= len(cCRE_pos):
                tmp = cCRE_pos[start:end]
                tmp = tmp[len(tmp)//2]
                cCRE_pos_new.append(tmp)
                start = end
        cCRE_set[chrom] = cCRE_pos_new
        print("The number of merged cCREs on {} is {}".format(chrom, len(cCRE_set[chrom])))
    # selecting cCREs around genes (TSS) according to their distances
    genes = {}
    with open(gene_file) as f:
        for line in f:
            line_split = line.strip().split('\t')
            chrom = line_split[0]
            start = int(line_split[1])
            end = int(line_split[2])
            strand = line_split[3]
            gene_type = line_split[4]
            gene_name = line_split[5]
            gene_id = line_split[6]
            expression = float(line_split[-1])
            if chrom not in INDEX:
                continue
            # appoint a gene type
            if gene_type != 'protein_coding':
                continue
            if strand == '+':
                TSS = start
            elif strand == '-':
                TSS = end
            else:
                print("no exact direction.")
                sys.exit(0)
            # compute the distances between genes' TSS and cCREs
            cCRE_pos = np.array(cCRE_set[chrom], dtype=np.int64)
            distance = cCRE_pos - TSS
            # the left direction
            left_index = (distance < 0)
            left_dis = distance[left_index]
            left_pos = cCRE_pos[left_index]
            index = np.argsort(left_dis)
            index = index[::-1]
            index_trim = index[:NUMBER]
            left_pos_retained = left_pos[index_trim]
            if len(left_pos_retained) < NUMBER:
                left_pos_retained = np.pad(left_pos_retained, (0, NUMBER-len(left_pos_retained)), mode='constant')
            left_pos_retained = left_pos_retained[::-1]
            # the right direction
            right_index = (distance > 0)
            right_dis = distance[right_index]
            right_pos = cCRE_pos[right_index]
            index = np.argsort(right_dis)
            index_trim = index[:NUMBER]
            right_pos_retained = right_pos[index_trim]
            if len(right_pos_retained) < NUMBER:
                right_pos_retained = np.pad(right_pos_retained, (0, NUMBER-len(right_pos_retained)), mode='constant')
            genes[gene_name] = {'info': [chrom, TSS, strand, expression, gene_id],
                                'cCRE_left_pos': left_pos_retained,
                                'cCRE_right_pos': right_pos_retained}
    return genes


def determine_index(gene_dict, target_gene_dict):
    gene_ccre = {}
    index_label_dict = {}
    for gene, loc_set in target_gene_dict.items():
        gene_name = gene
        if gene_name not in gene_dict.keys():
            print("the gene {} is not existed.".format(gene_name))
            continue
        chrom, TSS, _, expression, _ = gene_dict[gene_name]['info']
        left_pos_retained = gene_dict[gene_name]['cCRE_left_pos']
        right_pos_retained = gene_dict[gene_name]['cCRE_right_pos']
        pos_retained = np.concatenate((left_pos_retained, np.array([TSS]), right_pos_retained))
        #
        M = len(left_pos_retained)
        index_label_b = []
        index_label_m = []
        index_label_h = []
        index_label_s = []
        for loc in loc_set:
            mid = loc[1]
            label = loc[2]
            dis = np.abs(pos_retained-mid)
            index = np.argsort(dis)
            if index[0] == M:
                index_i = index[1]
            else:
                index_i = index[0]
            distance = dis[index_i] // 1000
            if distance > 1:
                continue
            # pos_retained[index_i] = mid
            distance = np.abs(mid-TSS)//1000
            if distance <= THRESHOLDS[0]:
                index_label_b.append((int(index_i), int(label)))
            elif THRESHOLDS[0] < distance <= THRESHOLDS[1]:
                index_label_m.append((int(index_i), int(label)))
            elif THRESHOLDS[1] < distance <= THRESHOLDS[2]:
                index_label_h.append((int(index_i), int(label)))
            elif distance > THRESHOLDS[2]:
                index_label_s.append((int(index_i), int(label)))
            else:
                sys.exit(0)
            
        gene_ccre[gene_name] = {'info': [chrom, TSS, expression],
                                'cCRE_left_pos': pos_retained[:M],
                                'cCRE_right_pos': pos_retained[(M+1):]}
        index_label_dict[gene_name] = {'index_label_b': index_label_b,
                                       'index_label_m': index_label_m,
                                       'index_label_h': index_label_h,
                                       'index_label_s': index_label_s}
    return gene_ccre, index_label_dict


def one_hot(seq, motif_dict):
    seq_dict = {'A':[1, 0, 0, 0], 'G':[0, 0, 1, 0],
                'C':[0, 1, 0, 0], 'T':[0, 0, 0, 1],
                'a':[1, 0, 0, 0], 'g':[0, 0, 1, 0],
                'c':[0, 1, 0, 0], 't':[0, 0, 0, 1]}
    temp = []
    for c in seq:
        temp.append(seq_dict.get(c, [0, 0, 0, 0]))
    temp = np.array(temp)
    motif_score = []
    # score
    def normalization(x):
        x_max = np.max(x)
        x_min = np.min(x)
        x_norm = (x - x_min) / (x_max - x_min)
        if (x_max - x_min) == 0:
            return x
        return np.round(x_norm, decimals=2)
    for key, pwm in motif_dict.items():
        score = convolve2d(temp, pwm, mode='valid')
        max_ = np.max(score.reshape(-1))
        motif_score.append(max_)
        
    return temp, normalization(np.array(motif_score))


def getbigwig(file, chrom, start, end):
    bw = pyBigWig.open(file)
    sample = np.array(bw.values(chrom, start, end))
    bw.close()
    return sample


def extract_track(data_dir, tracks, chrom, start, end):
    features = []
    for track in tracks:
        feature = getbigwig(data_dir + '/{}_merged.bigWig'.format(track), chrom, start, end)
        feature[np.isnan(feature)] = 0.
        # feature = np.log1p(feature)
        feature = np.log10(1 + feature)
        features.append(feature)

    return features


def retrieve_loop(distance, loop_dict, bin=5000):
    distance = np.array(distance, dtype=np.int64)
    distance = np.around(distance / bin)
    length = len(distance)
    interaction = np.zeros((length, length))
    # calculate interactions of all elements
    for i in range(length):
        fragment1_s = int(distance[i])
        for j in range(length):
            fragment2_s = int(distance[j])
            if i == j: continue
            if fragment1_s <= fragment2_s:
                key = '{}-{}'.format(fragment1_s, fragment2_s)
            else:
                key = '{}-{}'.format(fragment2_s, fragment1_s)
            if key in loop_dict.keys():
                value = loop_dict[key]
            else:
                value = 0
            interaction[i, j] = np.log10(1+value)
    
    return interaction


def encode_data(gene_set, motif_dict, sequence_dict, tracks, loop_dict, data_dir, out_f):
    gene_names_all, seqs_all, motifs_all, features_all, exp_all, dis_all, inter_all = [], [], [], [], [], [], []
    for key, value in gene_set.items():
        chrom, TSS, expression = value['info']
        cCRE_left_pos = value['cCRE_left_pos']
        cCRE_right_pos = value['cCRE_right_pos']
        # extract the gene and corresponding cCREs, and signals from tracks
        seqs = []
        motifs = []
        features = []
        distance = []
        num_left = 0
        num_right = 0
        # the cCREs on the left
        for mid in cCRE_left_pos:
            if mid == 0:
                cCRE = 'N' * SEQ_LEN
                seq, motif = one_hot(cCRE, motif_dict)
                seqs.append(seq)
                motifs.append(motif)
                feature = [np.zeros(SEQ_LEN) for i in range(len(tracks))]  # 4xL
                features.append(feature)
                num_left += 1
            else:
                start = mid - SEQ_LEN // 2
                end = mid + int(np.ceil(SEQ_LEN/2))
                cCRE = str(sequence_dict[chrom].seq[start:end])
                seq, motif = one_hot(cCRE, motif_dict)
                seqs.append(seq)
                cCRE_rc = ''
                for c in cCRE[::-1]:
                    cCRE_rc += seq_rc_dict.get(c, 'N')
                _, motif_rc = one_hot(cCRE_rc, motif_dict)
                motifs.append(np.maximum(motif, motif_rc))
                # motifs.append(motif)
                feature = extract_track(data_dir, tracks, chrom, start, end)  # 4xL
                features.append(feature)
                distance.append(mid)
        # the gene interval
        start = TSS - SEQ_LEN // 2
        end = TSS + int(np.ceil(SEQ_LEN/2))
        gene = str(sequence_dict[chrom].seq[start:end])
        seq, motif = one_hot(gene, motif_dict)
        seqs.append(seq)
        gene_rc = ''
        for c in gene[::-1]:
            gene_rc += seq_rc_dict.get(c, 'N')
        _, motif_rc = one_hot(gene_rc, motif_dict)
        motifs.append(np.maximum(motif, motif_rc))
        # motifs.append(motif)
        feature = extract_track(data_dir, tracks, chrom, start, end)  # 4xL
        features.append(feature)
        distance.append(TSS)
        # the cCREs on the right
        for mid in cCRE_right_pos:
            if mid == 0:
                cCRE = 'N' * SEQ_LEN
                seq, motif = one_hot(cCRE, motif_dict)
                seqs.append(seq)
                motifs.append(motif)
                feature = [np.zeros(SEQ_LEN) for i in range(len(tracks))]  # 4xL
                features.append(feature)
                num_right += 1
            else:
                start = mid - SEQ_LEN // 2
                end = mid + int(np.ceil(SEQ_LEN/2))
                cCRE = str(sequence_dict[chrom].seq[start:end])
                seq, motif = one_hot(cCRE, motif_dict)
                seqs.append(seq)
                cCRE_rc = ''
                for c in cCRE[::-1]:
                    cCRE_rc += seq_rc_dict.get(c, 'N')
                _, motif_rc = one_hot(cCRE_rc, motif_dict)
                motifs.append(np.maximum(motif, motif_rc))
                # motifs.append(motif)
                feature = extract_track(data_dir, tracks, chrom, start, end)  # 4xL
                features.append(feature)
                distance.append(mid)
        if len(distance) < 2*NUMBER+1:
            distance = np.pad(distance, (num_left, num_right), mode='edge')
        # extract loops from loop_dict
        interaction = retrieve_loop(distance, loop_dict[chrom])
        inter_all.append(interaction)
        gene_names_all.append(key.encode())
        seqs_all.append(seqs)
        motifs_all.append([motifs])
        features_all.append(features)
        exp_all.append([expression])
        dis_all.append(distance)
    # save data
    gene_names_all = np.array(gene_names_all)
    seqs_all = np.array(seqs_all)
    seqs_all = seqs_all.transpose((0, 3, 1, 2)) # N*4*M*L
    motifs_all = np.array(motifs_all) # N*1*M*L
    features_all = np.array(features_all)
    features_all = features_all.transpose((0, 2, 1, 3)) # N*4*M*L
    exp_all = np.array(exp_all)
    dis_all = np.array(dis_all)
    inter_all = np.array(inter_all)
    outputHDF5(gene_names_all, seqs_all, motifs_all, features_all, exp_all, dis_all, inter_all, out_f)


def outputHDF5(gene_names, seqs, motifs, features, exp, dis, inter, out_f):
    print('sequence shape: {}\tfeature shape: {}\tlabel shape: {}\n'.format(seqs.shape, features.shape, exp.shape))
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    dt = h5py.special_dtype(vlen=str)
    with h5py.File(out_f, 'w') as f:
        f.create_dataset('gene', data=gene_names,  dtype=dt, **comp_kwargs)
        f.create_dataset('seq', data=seqs, **comp_kwargs)
        f.create_dataset('motif', data=motifs, **comp_kwargs)
        f.create_dataset('fea', data=features, **comp_kwargs)
        f.create_dataset('exp', data=exp, **comp_kwargs)
        f.create_dataset('dis', data=dis, **comp_kwargs)
        f.create_dataset('loop', data=inter, **comp_kwargs)


def getPWM(motif_dir, motif_file):
    motif_dict = {}
    for x in motif_file:
        file = motif_dir + '/' + x
        pwm = []
        with open(file) as f:
            head = f.readline()
            head_split = head.strip().split('.')
            name = head_split[0][1:]
            for line in f:
                line_split = line.strip().split()
                pwm.append([float(i) for i in line_split])
        pwm = np.array(pwm)
        if name not in motif_dict:
            motif_dict[name] = pwm
        else:
            print(name)
    return motif_dict


def readloop(loopfile, bin=5000):
    loop_dict = {}
    with open(loopfile) as f:
        for line in f:
            line_split = line.strip().split('\t')
            chrom = line_split[0]
            if chrom not in loop_dict.keys():
                loop_dict[chrom] = {}
            fragment1_s = int(line_split[1]) // bin
            fragment2 = re.split(':|-|,', line_split[-1])
            fragment2_s = int(fragment2[1]) // bin
            value = float(fragment2[-1])
            if np.isinf(value):
                continue
            if fragment1_s <= fragment2_s:
                key = '{}-{}'.format(fragment1_s, fragment2_s)
            else:
                key = '{}-{}'.format(fragment2_s, fragment1_s)
            if key not in loop_dict[chrom]:
                loop_dict[chrom][key] = value
    return loop_dict


def normalization(x, flag=True):
    M = len(x) // 2
    x_part = np.concatenate((x[:M], x[(M+1):]))
    x_max = np.max(x_part)
    x_min = np.min(x_part)
    if flag:
        x_norm = (x - x_min) / (x_max - x_min)
    else:
        x_norm = (x - x_min) / (x_max - x_min) * 2 - 1
    x_norm[M] = 0
    return x_norm


def compute_attention(seqs, motif, fea, index_label, x, loop, flag=False):
    _, _, M, _ = seqs.size()
    M //= 2
    seqs_b = seqs[:,:,(M-x):(M+x+1),:]
    motif_b = motif[:,:,(M-x):(M+x+1),:]
    fea_b = fea[:,:,(M-x):(M+x+1),:]
    if loop == None:
        loop_b = None
    else:
        loop_b = loop[:,(M-x):(M+x+1),(M-x):(M+x+1)]
    # Load weights
    device = torch.device("cpu") 
    checkpoint_file = osp.join('/path/ScPGE/models/K562'.format(x), 'model.best.pth')
    chk = torch.load(checkpoint_file, map_location='cpu') # cuda:0
    state_dict = chk['model_state_dict']
    if flag: state_dict = MGPUtoSingle(state_dict)
    model = GenePredictionT5()
    model.load_state_dict(state_dict)
    model.to(device)
    # set eval state for Dropout and BN layers
    model.eval()
    with torch.no_grad():
        pred, attention = model(seqs_b, motif_b, fea_b, loop_b)
    attention = [att.data.numpy() for att in attention]
    ## integrate all attentions across all layers
    attention_all = []
    for step, att in enumerate(attention):
        if step == 0:
            attention_all = att[0] # np.abs(att[0])
        else:
            attention_all += att[0]
    _, m, _ = attention_all.shape
    att_a = np.mean(attention_all, axis=0) # m*m
    att_gene = att_a[m//2]
    att_gene = normalization(att_gene)
    att_gene[m//2] = 0
    score_all = []
    label_all = []
    for index, label in index_label:
        index_update = index - M + x
        if index_update < 0 or index_update >= m:
            continue
        score = att_gene[index_update]
        score_all.append(score)
        label_all.append(label)
    
    return score_all, label_all


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Compute the attention scores.")

    parser.add_argument("-r", dest="root", type=str, default="/path/ScPGE/data")
    parser.add_argument("-n", dest="name", type=str, default="K562")
    parser.add_argument("-c", dest="checkpoint", type=str, default="/path/ScPGE/models/K562")
    parser.add_argument("-t", dest="type", type=str, default='ScPGE', choices=['ScPGE', 'ScPGE-LP', 'ScPGE-KL'])

    return parser.parse_args()


def main():
    """Create the model and start the training."""
    args = get_args()
    target = args.name
    model_type = args.type
    device = torch.device("cpu") # cuda:0
    # load all motifs
    motif_dir = args.root + '/MOTIF/{}/pfm'.format(target)
    motif_file = os.listdir(motif_dir)
    motif_dict = getPWM(motif_dir, motif_file)
    data_dir = args.root + '/Human-RAWDATA/{}'.format(target)
    gene_file = data_dir + '/gene_expression.tsv'
    # load cis-regulatory elements
    cCRE_file = args.root + '/cCREs/hg38-cCREs.bed'
    gene_dict = gene_selection(gene_file, cCRE_file)
    source = 'Integration'
    crispr_file = args.root + '/CRISPR/{}.hg38.bed'.format(source)
    df = pd.read_csv(crispr_file, sep='\t')
    row, _ = df.shape
    target_gene = {}
    for i in range(row):
        chrom = df.loc[i]['chr']
        start = df.loc[i]['start']
        end = df.loc[i]['end']
        gene_name = df.loc[i]['Gene']
        label = df.loc[i]['Significant']
        mid = (start + end) // 2
        if gene_name not in target_gene.keys():
            target_gene[gene_name] = [(chrom, mid, label)]
        else:
            target_gene[gene_name] += [(chrom, mid, label)]
    
    target_gene_dict, index_label_dict = determine_index(gene_dict, target_gene)
    out_f = args.root + '/CRISPR/{}.json'.format(source)
    with open(out_f, 'w') as f:
        json.dump(index_label_dict, f)
    out_f = args.root + '/CRISPR/{}.hdf5'.format(source)
    genome = args.root + '/Genome'
    loop_file = args.root + '/Loops/{}/H3K27ac.5kb.longrange.bed'.format(target)
    loop_dict = readloop(loop_file)
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open(genome + '/hg38.fa'), 'fasta'))
    tracks = ['DNase', 'H3K4me3', 'H3K27ac', 'CTCF']
    encode_data(target_gene_dict, motif_dict, sequence_dict, tracks, loop_dict, data_dir, out_f)
    #### load data
    with h5py.File(out_f, 'r') as f:
        gene_names = np.array(f['gene'])
        gene_names = [x.decode() for x in gene_names]
        seq_te = np.array(f['seq'])
        motif_te = np.array(f['motif'])
        fea_te = np.array(f['fea'])
        exp_te = np.array(f['exp'])
        if model_type == 'ScPGE':
            te_loader = DataLoader(SourceDataSet(seq_te, motif_te, fea_te, exp_te),
                            batch_size=1, shuffle=False)
        else:
            loop_te = np.array(f['loop'])
            loop_te[np.isinf(loop_te)] = 0
            te_loader = DataLoader(SourceDataSetLoop(seq_te, motif_te, fea_te, exp_te, loop_te),
                            batch_size=1, shuffle=False)    
    ############ Classification of CRISPR-based data through attentions ##################
    bottom = [[],[]]
    mid = [[],[]]
    high = [[],[]]
    super = [[],[]]
    for i, sample_batch in enumerate(te_loader):
        seqs = sample_batch["seq"].float().to(device)
        motif = sample_batch["motif"].float().to(device)
        fea = sample_batch["fea"].float().to(device)
        if model_type == 'ScPGE':
            loop = None
        else:
            loop = sample_batch["loop"].float().to(device)
        gene_name = gene_names[i]
        # bottom
        x = 10
        index_label_b = index_label_dict[gene_name]['index_label_b']
        if len(index_label_b) > 0:
            score, label = compute_attention(seqs, motif, fea, index_label_b, x, loop, False)
            bottom[0] += score
            bottom[1] += label
        # middle
        x = 30
        index_label_m = index_label_dict[gene_name]['index_label_m']
        if len(index_label_m) > 0:
            score, label = compute_attention(seqs, motif, fea, index_label_m, x, loop, False)
            mid[0] += score
            mid[1] += label
        # high
        x = 50
        index_label_h = index_label_dict[gene_name]['index_label_h']
        if len(index_label_h) > 0:
            score, label = compute_attention(seqs, motif, fea, index_label_h, x, loop, True)
            high[0] += score
            high[1] += label
        # super
        x = 100
        index_label_s = index_label_dict[gene_name]['index_label_s']
        if len(index_label_s) > 0:
            score, label = compute_attention(seqs, motif, fea, index_label_s, x, loop, True)
            super[0] += score
            super[1] += label
    
    print("Running attention from GenePredictionT5\n")
    prauc = average_precision_score(bottom[1], bottom[0])
    num_pos = np.sum(np.asarray(bottom[1]) == 1)
    num_neg = np.sum(np.asarray(bottom[1]) == 0)
    print("No. of pos and neg enhance-gene pairs in bottom is {} and {}\n".format(num_pos, num_neg))
    print("PRAUC is {}.\n".format(prauc))
    #
    prauc = average_precision_score(mid[1], mid[0])
    num_pos = np.sum(np.asarray(mid[1]) == 1)
    num_neg = np.sum(np.asarray(mid[1]) == 0)
    print("No. of pos and neg enhance-gene pairs in mid is {} and {}\n".format(num_pos, num_neg))
    print("PRAUC is {}.\n".format(prauc))
    #
    prauc = average_precision_score(high[1], high[0])
    num_pos = np.sum(np.asarray(high[1]) == 1)
    num_neg = np.sum(np.asarray(high[1]) == 0)
    print("No. of pos and neg enhance-gene pairs in high is {} and {}\n".format(num_pos, num_neg))
    print("PRAUC is {}.\n".format(prauc))
    #
    prauc = average_precision_score(super[1], super[0])
    num_pos = np.sum(np.asarray(super[1]) == 1)
    num_neg = np.sum(np.asarray(super[1]) == 0)
    print("No. of pos and neg enhance-gene pairs in super is {} and {}\n".format(num_pos, num_neg))
    print("PRAUC is {}.\n".format(prauc))
    

if __name__ == "__main__":
    main()

