#!/bin/bash
# conda install bioconda::ucsc-bigwigmerge
# conda install bioconda::ucsc-bedgraphtobigwig
ROOT='/path/ScPGE'

# tracks=('H3K4me1' 'H3K4me2' 'H2AFZ' 'H3K36me3' 'H3K9ac' 'H3K9me3' 'H3K27me3' 'POLR2A' 'EP300')
tracks=('DNase' 'H3K4me3' 'H3K27ac' 'CTCF')

for target in $(ls ${ROOT}/data/Human-RAWDATA/)
do
    echo "Working on ${target}."
    # gene expression preparation
    python ${ROOT}/code/data_pre.py -r ${ROOT}/data -t ${target}
    for track in ${tracks[*]}
    do
        echo "merging ${track} tracks"
        bigWigMerge -max ${ROOT}/data/Human-RAWDATA/${target}/download/*.${track}.bigWig ${ROOT}/data/Human-RAWDATA/${target}/temp.bedGraph
        sort -k1,1 -k2,2n ${ROOT}/data/Human-RAWDATA/${target}/temp.bedGraph > ${ROOT}/data/Human-RAWDATA/${target}/temp_sorted.bedGraph
        bedGraphToBigWig ${ROOT}/data/Human-RAWDATA/${target}/temp_sorted.bedGraph ${ROOT}/data/Genome/hg38.chrom.sizes ${ROOT}/data/Human-RAWDATA/${target}/${track}_merged.bigWig

    done
    rm -f ${ROOT}/data/Human-RAWDATA/${target}/*.bedGraph
    
    # #
done


