#!/usr/bin/env python

import os, argparse
import os.path as osp


def get_args():
    parser = argparse.ArgumentParser(description="pre-process data.")
    parser.add_argument("-i", dest="inputfile", type=str, default='')
    parser.add_argument("-o", dest="outdir", type=str, default='')

    return parser.parse_args()


def download(inputfile, outdir, tracks):
    with open(inputfile) as f:
        lines = f.readlines()

    for line in lines:
        line_split = line.strip().split('\t')
        accession = line_split[0]
        format = line_split[1]
        type = line_split[2]
        assay = line_split[4].split(' ')[0]
        biosample = line_split[5]
        biosample_t = line_split[6]
        target = line_split[7].split('-')[0]
        replicate = line_split[-1]

        if 'transcript' in type:
            print("downloading transcripts from {}...".format(biosample))
            url = 'https://www.encodeproject.org/files/{}/@@download/{}.{}'.format(accession, accession, format)
            outfile = outdir + '/{}.transcript.{}'.format(accession, format)
            os.system('curl -o {} -J -L {}'.format(outfile, url))
        if 'DNase' in assay:
            print("downloading DNase-seq tracks from {}...".format(biosample))
            url = 'https://www.encodeproject.org/files/{}/@@download/{}.{}'.format(accession, accession, format)
            outfile = outdir + '/{}.DNase.{}'.format(accession, format)
            os.system('curl -o {} -J -L {}'.format(outfile, url))
        if target in tracks:
            print("downloading {} tracks from {}...".format(target, biosample))
            url = 'https://www.encodeproject.org/files/{}/@@download/{}.{}'.format(accession, accession, format)
            outfile = outdir + '/{}.{}.{}'.format(accession, target, format)
            os.system('curl -o {} -J -L {}'.format(outfile, url))


args = get_args()
tracks = ['H3K4me3', 'H3K27ac', 'CTCF']
# tracks = ['H3K4me1', 'H3K4me2', 'H2AFZ', 'H3K36me3', 'H3K9ac', 'H3K9me3', 'H3K27me3', 'POLR2A', 'EP300']
download(args.inputfile, args.outdir, tracks)




