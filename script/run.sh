#!/bin/bash

ROOT=/path/ScPGE
NUM=10

for target in $(ls ${ROOT}/data/Human-DATA${NUM} | grep -E "GM12878|K562")
do
    echo "Working on ${target}."
    if [ ! -d ${ROOT}/models/${target} ]; then
        mkdir -p ${ROOT}/models/${target}
    fi
    
    echo ">> Starting to warm up the model. <<"
    python ${ROOT}/code/warm_up.py -d ${ROOT}/data/Human-DATA${NUM}/${target} \
                                   -n ${target} \
                                   -c ${ROOT}/models/${target} \
                                   -t "ScPGE"
    echo ">> Warming up is finished. <<"

    echo ">> Starting to train the model. <<"
    python ${ROOT}/code/train.py -d ${ROOT}/data/Human-DATA${NUM}/${target} \
                                 -n ${target} \
                                 -c ${ROOT}/models/${target} \
                                 -t "ScPGE"
    echo ">> Training is finished. <<"
    # testing
    echo ">> Starting to test the model. <<"
    python ${ROOT}/code/test.py -d ${ROOT}/data/Human-DATA${NUM}/${target} \
                                -n ${target} \
                                -c ${ROOT}/models/${target} \
                                -t "ScPGE"
    echo ">> Testing is finished.<<"

done