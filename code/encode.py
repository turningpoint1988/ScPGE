# coding:utf-8
import os.path as osp
import os
import sys
import numpy as np
from Bio import SeqIO
import pyBigWig
import h5py
import argparse
from scipy.signal import convolve2d

SEQ_LEN = 600
WINDOW = 600
INDEX = ['chr' + str(i + 1) for i in range(23)]
INDEX[22] = 'chrX'
seq_rc_dict = {'A': 'T', 'G': 'C',
               'C': 'G', 'T': 'A',
               'a': 'T', 'g': 'C',
               'c': 'G', 't': 'A'}


def normalization(x):
    x_max = np.max(x)
    x_min = np.min(x)
    x_norm = (x - x_min) / (x_max - x_min)
    if (x_max - x_min) == 0:
        return x
    return np.round(x_norm, decimals=2)


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
        feature = np.log10(1 + feature)
        features.append(feature)

    return features


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


def datasplit(gene_set, motif_dict, sequence_dict, tracks, split, splice, data_dir, out_dir, status):
    gene_names_all, seqs_all, motifs_all, features_all, exp_all, dis_all = [], [], [], [], [], []
    gene_index = 0
    for key, value in gene_set.items():
        chrom, TSS, strand, expression, gene_id = value['info']
        if chrom not in split:
            continue
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
                feature = extract_track(data_dir, tracks, chrom, start, end)  # 4xL
                features.append(feature)
                distance.append(mid)
        if len(distance) < 2*NUMBER+1:
            distance = np.pad(distance, (num_left, num_right), mode='edge')
        # split data into tr, va, te
        gene_names_all.append(key.encode())
        seqs_all.append(seqs)
        motifs_all.append([motifs])
        features_all.append(features)
        exp_all.append([expression])
        dis_all.append(distance)
        gene_index += 1
        if gene_index % splice == 0:
            # save data
            gene_names_all = np.array(gene_names_all)
            seqs_all = np.array(seqs_all)
            seqs_all = seqs_all.transpose((0, 3, 1, 2)) # N*4*M*L
            motifs_all = np.array(motifs_all) # N*1*M*465
            features_all = np.array(features_all)
            features_all = features_all.transpose((0, 2, 1, 3)) # N*4*M*L
            exp_all = np.array(exp_all)
            dis_all = np.array(dis_all)
            out_f = out_dir + '/{}_{}.hdf5'.format(status, int(np.ceil(gene_index / splice)))
            outputHDF5(gene_names_all, seqs_all, motifs_all, features_all, exp_all, dis_all, out_f)
            # set default values
            gene_names_all, seqs_all, motifs_all, features_all, exp_all, dis_all = [], [], [], [], [], []
    if len(gene_names_all) > 0:
        # save data
        gene_names_all = np.array(gene_names_all)
        seqs_all = np.array(seqs_all)
        seqs_all = seqs_all.transpose((0, 3, 1, 2)) # N*4*M*L
        motifs_all = np.array(motifs_all) # N*1*M*L
        features_all = np.array(features_all)
        features_all = features_all.transpose((0, 2, 1, 3)) # N*4*M*L
        exp_all = np.array(exp_all)
        dis_all = np.array(dis_all)
        out_f = out_dir + '/{}_{}.hdf5'.format(status, int(np.ceil(gene_index / splice)))
        outputHDF5(gene_names_all, seqs_all, motifs_all, features_all, exp_all, dis_all, out_f)


def outputHDF5(gene_names, seqs, motifs, features, exp, dis, out_f):
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
    assert len(motif_dict.keys()) == SEQ_LEN, print("The number of motifs is wrong.")
    return motif_dict


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Encoding data.")

    parser.add_argument("-r", dest="root", type=str, default="/path/ScPGE/data")
    parser.add_argument("-t", dest="target", type=str, default="K562")
    parser.add_argument("-o", dest="out_dir", type=str, default=None)
    parser.add_argument("-n", dest="number", type=int, default=10)

    return parser.parse_args()

args = get_args()
NUMBER = args.number

def main():
    ROOT = args.root
    target = args.target
    # load all motifs
    motif_dir = args.root + '/MOTIF/{}/pfm'.format(target)
    motif_file = os.listdir(motif_dir)
    motif_dict = getPWM(motif_dir, motif_file)
    genome = ROOT + '/Genome'
    data_dir = ROOT + '/Human-RAWDATA/{}'.format(target)
    cCRE_dir = ROOT + '/cCREs'
    out_dir = args.out_dir
    #
    gene_file = data_dir + '/gene_expression.tsv'
    cCRE_file = cCRE_dir + '/hg38-cCREs.bed'
    gene_set = gene_selection(gene_file, cCRE_file)
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open(genome + '/hg38.fa'), 'fasta'))
    tracks = ['DNase', 'H3K4me3', 'H3K27ac', 'CTCF']
    split_te = ['chr8', 'chr9']
    split_va = ['chr16']
    #
    splice = 2000
    datasplit(gene_set, motif_dict, sequence_dict, tracks, split_va, splice, data_dir, out_dir, 'va')
    #
    datasplit(gene_set, motif_dict, sequence_dict, tracks, split_te, splice, data_dir, out_dir, 'te')
    #
    for x in (split_te + split_va):
        INDEX.remove(x)
    split_tr = INDEX
    datasplit(gene_set, motif_dict, sequence_dict, tracks, split_tr, splice, data_dir, out_dir, 'tr')


if __name__ == '__main__':  main()
