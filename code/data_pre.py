# coding:utf-8
import os.path as osp
import os
import glob
import numpy as np
import argparse


def extract_gene(gencode_file):
    genes = {}
    with open(gencode_file) as f:
        lines = f.readlines()

    for line in lines[5:]:
        line_split = line.strip().split('\t')
        chrom = line_split[0]
        type = line_split[2]
        start = line_split[3]
        end = line_split[4]
        strand = line_split[6]
        info = line_split[8]
        info_split = info.split(';')
        gene_id = eval(info_split[0].strip().split()[-1])
        if type == 'gene' and gene_id not in genes.keys():
            gene_type = eval(info_split[1].strip().split()[-1])
            gene_name = eval(info_split[2].strip().split()[-1])
            head = '{}\t{}\t{}\t{}\t{}\t{}'.format(chrom, start, end, strand, gene_type, gene_name)
            genes[gene_id] = [head]
        if type == 'transcript':
            transcript_id = eval(info_split[1].strip().split()[-1])
            genes[gene_id] += [transcript_id]

    return genes


def extract_transcript(transcript_file):
    genes = {}
    with open(transcript_file) as f:
        lines = f.readlines()
    head = lines[0].strip().split()
    if len(head) == 5:
        for line in lines[1:]:
            line_split = line.strip().split()
            info = line_split[0]
            tpm = float(line_split[-1])
            info_split = info.split('|')
            if len(info_split) < 2:
                continue
            gene_id = info_split[1]
            if gene_id not in genes.keys():
                genes[gene_id] = [tpm]
            else:
                genes[gene_id] += [tpm]
    else:
        for line in lines[1:]:
            line_split = line.strip().split()
            gene_id = line_split[1]
            tpm = float(line_split[5])
            # ENSM; ENSG
            if 'ENSM' not in gene_id:
                continue
            if gene_id not in genes.keys():
                genes[gene_id] = [tpm]
            else:
                genes[gene_id] += [tpm]

    return genes


def integrate_gene(transcript_files, genes, out_f):
    transcripts = []
    for each_file in transcript_files:
        transcripts.append(extract_transcript(each_file))

    f = open(out_f, 'w')
    for key, value in genes.items():
        gene_id = key
        info = value[0]
        transcript_id = value[1:]
        expression = []
        for transcript in transcripts:
            if gene_id not in transcript.keys():
                continue
            expression += transcript[gene_id]
        assert len(transcript_id)*len(transcripts) == len(expression)
        expression = np.log10(1 + np.sum(expression))
        f.write("{}\t{}\t{:.3f}\n".format(info, gene_id, expression))

    f.close()


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Data preparation")

    parser.add_argument("-r", dest="root", type=str, default="/path/ScPGE/data")
    parser.add_argument("-t", dest="target", type=str, default="K562")

    return parser.parse_args()


def main():
    args = get_args()
    ROOT = args.root
    target = args.target
    # extract genes from gencode annotation
    # gencode.vM21.annotation.gtf; gencode.v29.annotation.gtf
    gencode = ROOT + '/GENCODE/gencode.v29.annotation.gtf'
    genes = extract_gene(gencode)
    # integrate multiple transcripts from the same cell line/type
    transcript_files = glob.glob(ROOT + '/Human-RAWDATA/{}/download/*.transcript.tsv'.format(target))
    out_f = ROOT + '/Human-RAWDATA/{}/gene_expression.tsv'.format(target)
    integrate_gene(transcript_files, genes, out_f)


if __name__ == '__main__':  main()
