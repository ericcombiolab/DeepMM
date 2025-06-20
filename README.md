# DeepMM: Identify and correct Metagenome Misassemblies with deep learning
[![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fericcombiolab%2FDeepMM&label=Views&labelColor=%23697689&countColor=%23ff8a65&style=flat)](https://github.com/ericcombiolab/DeepMM)

## Requirements and Installation
Make sure you have the dependencies below installed and accessible in your $PATH.

### Prepare dependencies

- tqdm
- torch
- scipy 
- numpy
- pysam
- pandas
- biopython
- matplotlib
- torchvision
- scikit-learn
  
The above packages can be installed with DeepMM

### Other dependencies
- [bwa](https://github.com/lh3/bwa)
- [samtools](https://github.com/samtools/samtools?tab=readme-ov-file)
  
The above packages are necessary for data preparation.


### Installation

##### Install DeepMM via git

```
git clone https://github.com/ericcombiolab/DeepMM.git
```
- install

```
cd DeepMM
pip install .
DeepMM -h
```

## Quick Start
- Alignments

```
bwa index $contigs.fa
bwa mem -a -t 128 $contigs.fa $R1.fq.gz $R2.fq.gz > $align.sam
samtools faidx $contigs.fa
samtools view -h -q 10 -m 50 -F 4 -b $align.sam > $align.bam
samtools sort $align.bam -o $sort.bam
samtools index $sort.bam
```
Here we provide two model weight with different read profile (insert_size_mean, insert_size_standard_deviation)
- default: (200, 20)
  ```
  mkdir -p ./pretrain-model-weight
  wget -P ./pretrain-model-weight https://zenodo.org/records/15702546/files/checkpoint.pt
  ```
- CAMI-type: (270, 27)
  ```
  mkdir -p ./finetune-model-weight 
  wget -P ./finetune-model-weight https://zenodo.org/records/15702546/files/cami_finetune_checkPoint.pt
  ```

- Run DeepMM  
``` misassembly identification and correction;
DeepMM correct --folder_path /path/to/your/data\
               --model_weight_path /path/to/model\
               --fine_tune\ ##(only pass the parameter when you use finetuned model weight)
               --contig_threads 8\
               --gpu_device cuda:0
```
## Example

Files in `./example`

- `final.contigs.fa`
- `final.contigs.fai`
- `sort.bam`
- `sort.bam.bai`

```
DeepMM correct --folder_path ./example/final.contigs.fa \
               --model_weight_path ./finetune-model-weight/cami_finetune_checkPoint.pt \
               --fine_tune \
               --contig_threads 8 \
               --gpu_device cuda:0
```
The output is under `./example/deepmm_output`

## Supervised Fine-Tuning on new datasets (Highly recommend)
As reads from different sequencers have slight differernt, which may effect model performance. We highly recommand you fine-tuning model though cause some time, but worth to do. Only one community dataset could finish SFT.
### Software
- [MGSIM](https://github.com/nick-youngblut/MGSIM)
- [MEGAHIT](https://github.com/voutcn/megahit) or [metaSPAdes](https://github.com/ablab/spades)
- [metaQusat](https://quast.sourceforge.net/metaquast)
### Download reference
```
wget https://zenodo.org/records/15702546/files/reference.tar.gz
wget https://zenodo.org/records/15702546/files/genome_list.tsv
```
### Create read error profile
```
art_profiler_illumina error_profile_ path/to/your/read_1/file fastq
art_profiler_illumina error_profile_ path/to/your/read_2/file fastq
```
this can creat two read file in current directory
```
error_profile_R1.txt
error_profile_R2.txt
```
### Fine-tune Dataset simulation
```
comm_num=1

mkdir -p  ./simulation/communities

MGSIM communities --abund-dist=lognormal \
                  --abund-dist-p=mean:10,sigma:1 \
                  --n-comm=$comm_num \
                  ./genome_list.tsv \
                  ./simulation/communities/comm

#Tips: --art-sdev can# be set 10% of --art-mflen
MGSIM reads --art-mflen $your_sequencer_insert_size_mean\
            --art-sdev $your_sequencer_insert_size_standard_deviation\
            --art-len 150\
            --art-paired\
            --tmp-dir ./simulation/temp_reads\
            --sr-seq-depth 1e7 --rndSeed 8294 -n 128 --gzip\
            ./genome_list.tsv\
            --art-qprof1 ./error_profile_R1.txt\
            --art-qprof2 ./error_profile_R2.txt\
            ./simulation/communities/comm_wAbund.txt\
            ./simulation/reads

for ((i = 1; i <= $comm_num; i++))
do

    mkdir -p ./simulation/$i

    # Assemble (Based on your data assembler)
    ./bin/megahit -1 ./simulation/reads/illumina/$i/R1.fq.gz\
                  -2 ./simulation/reads/illumina/$i/R2.fq.gz\
                  -t 128\
                  --k-list 21,29,39,59,79,99\
                  --min-contig-len 1000\
                  -o ./simulation/$i/contigs

    # Assembly Evaluation
    python metaquast.py -t 128 \
                        --extensive-mis-size 100 \
                        --min-contig 1000\
                        ./simulation/$i/contigs/final.contigs.fa\
                        -r ./reference\
                        -o ./simulation/$i/metaquast

    # Alignment
    mkdir -p ./simulation/$i/align
    bwa index ./simulation/$i/contigs/final.contigs.fa
    bwa mem -a -t 128 ./simulation/$i/contigs/final.contigs.fa \
                      ./simulation/reads/illumina/$i/R1.fq.gz \
                      ./simulation/reads/illumina/$i/R2.fq.gz \
                    > ./simulation/$i/align/align.sam
    samtools faidx ./simulation/$i/contigs/final.contigs.fa
    samtools view -h -q 10 -m 50 \
                  -F 4 -b ./simulation/$i/align/align.sam \
                        > ./simulation/$i/align/align.bam
    samtools sort ./simulation/$i/align/align.bam \
               -o ./simulation/$i/align/sort.bam
    samtools index ./simulation/$i/align/sort.bam

    # Collect data
    mkdir -p ./simulation/data/$i
    cp -r ./simulation/$i/contigs/final.contigs.fa \
          ./simulation/data/$i/
    cp -r ./simulation/$i/contigs/final.contigs.fa.fai \
          ./simulation/data/$i/
    cp -r ./simulation/$i/align/sort.bam \
          ./simulation/data/$i/
    cp -r ./simulation/$i/align/sort.bam.bai \
          ./simulation/data/$i/
    cp -r ./simulation/$i/metaquast/combined_reference/contigs_reports/all_alignments_final-contigs.tsv \
          ./simulation/data/$i/

    echo 'Community' $i 'done'
done
```
### Fine-tune feature dataset

```
DeepMM finetune_data --data_folder ./simulation/data \
                     --feature_folder ./simulation/feature \
                     --label all_alignments_final-contigs.tsv \
                     --assembly final.contigs.fa \
                     --alignment sort.bam \
```

### Fine-tune
#### Download pretrained model weight
```
wget -P ./pretrain-model-weight https://zenodo.org/records/14863005/files/checkPoint.pt
```

```
DeepMM finetune --epochs 50 \
                --batch_size 256 \
                --lr 0.001 \
                --gpu_device cuda:0 \
                --clip 1 \
                --weight_decay 0.0001 \
                --checkPoint_path ./fine-tune-model-weight \
                --pretrain_model_path ./pretrain-model-weight/checkpoint.pt \
                --finetune_dataset_path ./simulation/feature
```
## Pretraining on new datasets
You can simulate your own pretrain datasets with differe settings. The dataset simulation processing is similar with `Fine-tuning` seciton. However, this is time-consuming, and fine-tuning can help model already.

```
DeepMM pretrain_data --data_folder /path/to/data \
                     --feature_folder /path/to/data \
                     --label all_alignments_final-contigs.tsv \
                     --assembly final.contigs.fa \
                     --alignment sort.bam \

DeepMM pretrain \
        --pretrain_dataset_path 'path/to/pretrain/dataset' \
        --eval_dataset_path 'path/to/evaluation/dataset' \ # This dataset can generated by finetune dataset module
        --epochs 50 \
        --batch-size 64 \
        --gpus $GPU_NUMBERS \
        --lr 0.003 \
        --checkPoint_path ./pretiran-model-weight
```
### DataParallel
```
 #!/bin/bash

export OMP_NUM_THREADS=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

GPU_NUMBERS=4

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
        --nproc_per_node=$GPU_NUMBERS \
        /DeepMM/DeepMM/ddp_pretrain.py \ # Local DeemMM file
        --pretrain_dataset_path 'path/to/pretrain/dataset' \
        --eval_dataset_path 'path/to/evaluation/dataset' \ # This dataset can generated by finetune dataset module
        --epochs 50 \
        --batch-size 64 \
        --gpus $GPU_NUMBERS \
        --lr 0.003 \
        --checkPoint_path ./pretiran-model-weight

```



## Output
The output folder ``deepmm_output`` will contain
1. Misassembly prediction for each contig:  
**assembly_prediction.tsv**   
Each column:
     - Assembly: Assembly id
     - BreakPoint: Misassembly breakPoint location with max prediction
     - Prediction: Max misassembly prediction score
     - Length: Assembly length
     - Chimeric BreakPoint: Chimeric misassembly brekpoint location with max prediction
     - Chimeric Prediction: Max chimeric misassembly prediction
1. Corrected assembly file:  
**corrected_contigs.fa**

# Complete option list
## 1. correct

Used to identify and correct misassemblies.

### Parameters:
- `--folder_path` (str, default=''): Path to data folder.
- `--model_weight_path` (str, default='./pretrain-model-weight/checkPoint.pt'): Path to model weight.
- `--fine_tune` (action='store_true'): Use fine-tune model.
- `-t`, `--threads` (int, default=8): Maximum number of threads [default: 8].
- `--gpu_device` (str, default='cuda:0'): GPU device ID.

## 2. pretrain

Used to pretrain the model.

### Parameters:
- `--eval_dataset_path` (str, metavar='DIR', default=''): Path to eavl dataset.
- `--pretrain_dataset_path` (str, metavar='DIR', default=''): Path to dataset.
- `-a`, `--arch` (str, metavar='ARCH', default='resnet50'): Model architecture: resnet18 | resnet50 (default: resnet50).
- `-j`, `--workers` (int, default=32, metavar='N'): Number of data loading workers (default: 32).
- `--epochs` (int, default=100, metavar='N'): Number of total epochs to run.
- `-b`, `--batch-size` (int, default=64, metavar='N'): Mini-batch size (default: 64).
- `--lr`, `--learning-rate` (float, default=0.0003, metavar='LR', dest='lr'): Initial learning rate.
- `--wd`, `--weight-decay` (float, default=1e-4, metavar='W', dest='weight_decay'): Weight decay (default: 1e-4).
- `--seed` (int, default=42): Seed for initializing training.
- `--out_dim` (int, default=128): Feature dimension (default: 128).
- `--temperature` (float, default=0.07): Softmax temperature (default: 0.07).
- `--n_views` (int, default=5, metavar='N'): Number of views for contrastive learning training.
- `--gpus` (int, default=1): Number of GPUs.
- `--checkPoint_path` (str, default='./pretiran-model-weight/checkPoint_{formatted_time}'): Checkpoint path.
- `--fp16-precision` (action='store_true'): Whether or not to use 16-bit precision GPU training.

## 3. finetune

Used to fine-tune the model.

### Parameters:
- `--epochs` (int, default=50): Number of epochs to run.
- `--batch_size` (int, default=256): Batch size.
- `--lr` (float, default=0.0001): Learning rate.
- `--gpu_device` (str, default='cuda:0'): GPU device ID.
- `--clip` (int, default=1): Gradient clipping value.
- `--weight_decay` (float, default=0.0001): Weight decay.
- `--checkPoint_path` (str, default='./fine-tune-model-weight'): Checkpoint path.
- `--pretrain_model_path` (str, default='./pretrain-model-weight/checkpoint.pt'): Pretrained model path.
- `--finetune_dataset_path` (str, default=''): Fine-tune dataset path.

## 4. pretrain_data

Used to generate pretrain dataset.

### Parameters:
- `--data_folder` (str, default=''): Path to data folder.
- `--feature_folder` (str, default=''): Path to feature folder.
- `--label` (str, default='all_alignments_final-contigs.tsv'): Label file name.
- `--assembly` (str, default='final.contigs.fa'): Assembly file name.
- `--alignment` (str, default='sort.bam'): Alignment file name.
- `--pretrain_dataset_path` (str, default=''): Path to pretrain dataset.

## 5. finetune_data

Used to generate finetune dataset.

### Parameters:
- `--data_folder` (str, default=''): Path to data folder.
- `--feature_folder` (str, default=''): Path to feature folder.
- `--label` (str, default='all_alignments_final-contigs.tsv'): Label file name.
- `--assembly` (str, default='final.contigs.fa'): Assembly file name.
- `--alignment` (str, default='sort.bam'): Alignment file name.
