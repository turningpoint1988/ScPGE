# ScPGE

**A scalable computational framework for predicting gene expression from candidate cis-regulatory elements.** <br/>
The flowchart of NLDNN-AT is displayed as follows:

<p align="center"> 
<img src=https://github.com/turningpoint1988/ScPGE/picture/flowchart.jpg>
</p>

<h4 align="center"> 
Fig.1 The flowchart of ScPGE.
</h4>

## Prerequisites and Dependencies

- [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
- Python 3.7
- [PyTorch 1.9](https://pytorch.org/)
- [CUDA 11.1](https://developer.nvidia.com/cuda-11.1.1-download-archive)
- Python packages: biopython, scikit-learn, pyBigWig, h5py, scipy, pandas, matplotlib, seaborn

## Other Tools

- [MEME Suite](https://meme-suite.org/meme/doc/download.html): It assembles several methods used by this paper, including MEME-ChIP, TOMTOM and FIMO.
- [Bedtools](https://bedtools.readthedocs.io/en/latest/content/installation.html): It is a powerful toolset for genome arithmetic.
- [Captum](https://github.com/pytorch/captum): It is a model interpretability and understanding library for PyTorch

## Competing Methods

- [Enformer](https://github.com/google-deepmind/deepmind-research/tree/master/enformer)
- [CREaTor](https://github.com/DLS5-Omics/CREaTor)
- [EPInformer](https://github.com/pinellolab/EPInformer)

## Data Preparation

- Download [hg38.fa](https://hgdownload.soe.ucsc.edu/downloads.html#human) and [mm10.fa](https://hgdownload.soe.ucsc.edu/downloads.html#mouse), and then put them into the `Genome` directory.
- Download [Human V29 annotation file](https://www.gencodegenes.org/human/release_29.html) and [Mouse M21 annotation file](https://www.gencodegenes.org/mouse/release_M21.html), and then put them into the `GENCODE` directory.
- Download [Experimental datasets](https://www.encodeproject.org) by using the following script:

```
cd /path/ScPGE/script
bash download.sh
```

After finished, you can run the following script to prepare data:

```
cd /path/ScPGE/script
bash data_pre.sh
```

## Data Construction 

 ScPGE assembles DNA sequences, TF binding scores, and epigenomic tracks from discrete cCREs into three 3-dimensional tensors and transforms chromatin loops into a 2-dimensional interaction matrix by using the following script: 

```
cd /path/ScPGE/script
bash encode.sh
```

## Running ScPGE

We can run ScPGE from the scratch using the following script:

```
cd /path/ScPGE/script
bash run.sh
```

This script includes three stages, (1) a ‘warm-up’ process: select the best-initialized model; (2) a training process: train a final model; (3) a testing process:  test the final model.



## Predictive Performance

RNA-seq and CAGE-seq gene expressions were used to to investigate the predictive performance of ScPGE.

<p align="center"> 
<img src=https://github.com/turningpoint1988/NLDNN/blob/main/picture/performance.jpg width = "600" height = "500">
</p>

<h4 align="center"> 
Fig.2 The performance of ScPGE in predicting RNA-seq/CAGE gene expression levels.
</h4>

## Pattern Discovery

Trough categorization of predictions, different patterns were found in true positives (TPs), false positives (FPs), true negatives (TNs), and false negatives (FNs).

<p align="center"> 
<img src=https://github.com/turningpoint1988/NLDNN/blob/main/picture/pattern.jpg>
</p>

<h4 align="center"> 
Fig.3 The patterns found in TPs, FPs, TNs, and FNs.
</h4>
