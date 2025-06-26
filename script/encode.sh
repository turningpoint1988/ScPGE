#!/bin/bash

ROOT=/path/ScPGE

threadnum=2
tmp="/tmp/$$.fifo"
mkfifo ${tmp}
exec 6<> ${tmp}
rm ${tmp}
for((i=0; i<${threadnum}; i++))
do
    echo ""
done >&6

NUM=10

for target in $(ls ${ROOT}/data/Human-RAWDATA/ | grep -E "GM12878|K562")
do
  read -u6
  {  
    echo "Working on ${target}." 
    if [ ! -d ${ROOT}/data/Human-DATA${NUM}/${target} ]; then
        mkdir -p ${ROOT}/data/Human-DATA${NUM}/${target}
    fi
    python ${ROOT}/code/encode.py -r ${ROOT}/data -t ${target} \
                             -o ${ROOT}/data/Human-DATA${NUM}/${target} \
                             -n ${NUM}

    echo "" >&6
   }&
done
wait
exec 6>&-



