#!/usr/bin/bash

ROOT='/path/ScPGE'

CELL=('GM12878' 'HCT116' 'A673' 'keratinocyte' 'PC-3' 'fibroblast of dermis' 'MCF-7' 'OCI-LY7' 'B cell' \
       'neural progenitor cell' 'bipolar neuron' 'HeLa-S3' 'Panc1' 'HepG2' 'K562' 'H1' \
       'CD14-positive monocyte' 'GM23338' 'astrocyte')
# CELL=('MEL' 'CH12.LX')

for cell in ${CELL[*]}
do
    
    if [ ! -d ${ROOT}/data/Human-RAWDATA/${cell} ]; then
            mkdir -p ${ROOT}/data/Human-RAWDATA/${cell}/download
    fi
    
    python ${ROOT}/code/download.py -i ${ROOT}/data/rawdata_list/"${cell}_list.txt" -o ${ROOT}/data/Human-RAWDATA/${cell}/download

done
