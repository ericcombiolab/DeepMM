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
  
The above packages can be installed with DeepMM.

Tested environment: Python 3.9 with torch 2.5.1+cu121, torchvision 0.20.1+cu121, pysam 0.24.0, biopython 1.85, scikit-learn 1.6.1, pandas 2.3.3, scipy 1.13.1, matplotlib 3.9.4, numpy 2.0.2, tqdm 4.70.0.

### Other dependencies
- [bwa](https://github.com/lh3/bwa)
- [samtools](https://github.com/samtools/samtools?tab=readme-ov-file)
  
The above packages are necessary for data preparation.


### Installation

### Install

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

### 1) Prepare alignments

```
bwa index $contigs.fa
bwa mem -a -t 128 $contigs.fa $R1.fq.gz $R2.fq.gz > $align.sam
samtools faidx $contigs.fa
samtools view -h -q 10 -m 50 -F 4 -b $align.sam > $align.bam
samtools sort $align.bam -o $sort.bam
samtools index $sort.bam
```

The data folder passed to DeepMM should contain `final.contigs.fa` and `sort.bam`.

### 2) Download the model weight

```
mkdir -p ./pretrain-model-weight
wget -P ./pretrain-model-weight https://github.com/ericcombiolab/DeepMM/releases/latest/download/checkpoint.pt
```

### 3) Run DeepMM

```
# predict: window inference + label-free E-value calibration (final per-contig score)
DeepMM predict --folder_path /path/to/your/data \
               --model_weight_path ./pretrain-model-weight/checkpoint.pt

# correct: split misassembled contigs (uses the calibrated predictions by default)
DeepMM correct --folder_path /path/to/your/data
```

## Example

Files in `./example`

- `final.contigs.fa`
- `final.contigs.fai`
- `sort.bam`
- `sort.bam.bai`

```
DeepMM predict --folder_path ./example \
               --model_weight_path ./pretrain-model-weight/checkpoint.pt \
               --contig_threads 8 \
               --gpu_device cuda:0

DeepMM correct --folder_path ./example
```
The output is under `./example/deepmm_output`: the per-contig prediction output `assembly_prediction.escore.tsv` (calibrated score `S = -log10(p)`, the score used by `correct` and in the paper) and the corrected assembly `corrected_contigs.fa`.

## Supervised Fine-Tuning on new datasets (Highly recommend)
As reads from different sequencers have slight different, which may effect model performance. We highly recommend you fine-tuning model though cause some time, but worth to do. Only one community dataset could finish SFT.
### Software
- [MGSIM](https://github.com/nick-youngblut/MGSIM)
- [MEGAHIT](https://github.com/voutcn/megahit) or [metaSPAdes](https://github.com/ablab/spades)
- [metaQUAST](https://quast.sourceforge.net/metaquast)
### Prepare genome list
Use your own reference genomes for simulation: provide a `genome_list.tsv` listing their FASTA paths (as used in the MGSIM commands below).
### Create read error profile
```
art_profiler_illumina error_profile_ path/to/your/read_1/file fastq
art_profiler_illumina error_profile_ path/to/your/read_2/file fastq
```
this can create two read file in current directory
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

#Tips: --art-sdev can be set 10% of --art-mflen
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
wget -P ./pretrain-model-weight https://github.com/ericcombiolab/DeepMM/releases/latest/download/checkpoint.pt
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
You can simulate your own pretrain datasets with differe settings. The dataset simulation processing is similar with `Fine-tuning` section. However, this is time-consuming, and fine-tuning can help model already.

```
DeepMM pretrain_data --data_folder /path/to/data \
                     --feature_folder /path/to/data \
                     --label all_alignments_final-contigs.tsv \
                     --assembly final.contigs.fa \
                     --alignment sort.bam \

python -m DeepMM.ddp_pretrain \
        --pretrain_dataset_path 'path/to/pretrain/dataset' \
        --eval_dataset_path 'path/to/evaluation/dataset' \ # This dataset can generated by finetune dataset module
        --epochs 50 \
        --batch-size 64 \
        --gpus $GPU_NUMBERS \
        --lr 0.003 \
        --checkPoint_path ./pretrain-model-weight
```
### DataParallel
```
 #!/bin/bash

export OMP_NUM_THREADS=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

GPU_NUMBERS=4

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
        --nproc_per_node=$GPU_NUMBERS \
        /DeepMM/DeepMM/ddp_pretrain.py \ # Local DeepMM file
        --pretrain_dataset_path 'path/to/pretrain/dataset' \
        --eval_dataset_path 'path/to/evaluation/dataset' \ # This dataset can generated by finetune dataset module
        --epochs 50 \
        --batch-size 64 \
        --gpus $GPU_NUMBERS \
        --lr 0.003 \
        --checkPoint_path ./pretrain-model-weight

```



## Output
The output folder `deepmm_output` contains:

1. **assembly_prediction.escore.tsv** — the per-contig prediction output: for every contig, the label-free calibrated significance score S = -log10(p), computed against an empirical null built from the same sample. Columns: `Assembly` (contig id), `BreakPoint` (predicted junction position; -1 below the E-value gate), `Prediction` (the calibrated score S), `Length`, `Chimeric_BreakPoint`, `Chimeric_Prediction` (interspecies-specific heuristic, kept for reference). This `Prediction` value is the final DeepMM score, used by `correct` and quoted in the paper.

2. **corrected_contigs.fa** — the input assembly with every selected contig split once at its predicted breakpoint (fragments suffixed `_1`, `_2`); contigs below the cutoff are copied unchanged.

(The uncalibrated window-maximum table `assembly_prediction.tsv` is kept only as an intermediate artifact; it is not the DeepMM output.)

# Complete option list

## 1. predict
`DeepMM predict` — identify misassemblies (inference only). The label-free E-value calibration is always applied, and its calibrated score is the final per-contig output.
- `--folder_path` (default=''): path to the data folder (contig FASTA + sorted BAM).
- `--model_weight_path` (default='./pretrain-model-weight/checkpoint.pt'): path to the model weight.
- `--fine_tune` (flag; not needed for the released weight): use a fine-tuned checkpoint.
- `--contig_threads` (int, 8), `--window_threads` (int, 8): CPU threads for extraction / window processing.
- `--gpu_device` (default='cuda:0'): GPU device id.
- `--file_name` (assembly_prediction.tsv): raw prediction TSV name.
- `--contig_file_name` (final.contigs.fa), `--bam_file_name` (sort.bam).
- `--threshold` (0.9): window-score cutoff used for the BreakPoint column.
- `--infer_batch_size` (32), `--contig_chunk_size` (8), `--max_windows_per_chunk` (1024): GPU streaming options.
- `--max_contig_length` (inf): maximum contig length to process.
- `--output_dir` (default: `<folder_path>/deepmm_output`), `--checkpoint_file` (checkpoint_predictions.jsonl), `--resume` / `--no_resume` (default: resume).
- The calibrated score is written next to the run as `assembly_prediction.escore.tsv` (`--escore_file_name`); `--escore_mode`, `--escore_r` and `--escore_floor` tune the empirical null.

## 2. correct
`DeepMM correct` — split misassembled contigs using the calibrated per-contig predictions (no inference).
- `--folder_path` (default=''): data folder containing the contig FASTA.
- `--contig_file_name` (final.contigs.fa); `--bam_file_name` (unused, kept for CLI compatibility).
- `--output_dir` (default: `<folder_path>/deepmm_output`): folder holding the prediction TSV and receiving corrected_contigs.fa.
- `--threshold`: splitting cutoff on the calibrated `Prediction` score; defaults to -log10(0.05). Keep it at or above the default, otherwise contigs can be selected while their BreakPoint is still -1 and are then dropped.

## 3. escore
`DeepMM escore` — utility that recomputes the calibrated per-contig score (S = -log10 p) from an existing run.
- `--folder_path`, `--output_dir`; `--file_name` (assembly_prediction.tsv); `--out_file_name` (default assembly_prediction.escore.tsv); `--in_place`.
- `--mode` (calibrated); `--r`; `--floor` / `--no_floor`; `--abstain_score` (0.0); `--bp_min_e` (default -log10(0.05); keep at or below the threshold passed to `correct`).

## 4. finetune
`DeepMM finetune` — supervised fine-tuning on your own communities/features (optional).
- `--epochs` (50), `--batch_size` (256), `--lr` (0.0001), `--clip` (1), `--weight_decay` (0.0001), `--gpu_device` (cuda:0).
- `--checkPoint_path` (./fine-tune-model-weight), `--pretrain_model_path` (./pretrain-model-weight/checkpoint.pt), `--finetune_dataset_path`.
- `--unfreeze` (linear|layer4|layer3_layer4|all; default layer4), `--balanced_sampler`, `--selection_metric` (loss|auprc; default auprc).

## 5. pretrain_data
`DeepMM pretrain_data` — generate a pretraining dataset.
- `--data_folder`, `--feature_folder`, `--label` (all_alignments_final-contigs.tsv), `--assembly` (final.contigs.fa), `--alignment` (sort.bam), `--pretrain_dataset_path`.

## 6. finetune_data
`DeepMM finetune_data` — generate a fine-tuning dataset.
- `--data_folder`, `--feature_folder`, `--label`, `--assembly`, `--alignment`.
- `--eval_only`; `--data_workers` (8); `--community_split_manifest` (split at whole-community level); `--balance_output`.

The pretraining loop itself is run with the DDP script: `python -m DeepMM.ddp_pretrain --pretrain_dataset_path ... --eval_dataset_path ... --epochs 50 --batch-size 64 --lr 0.003 --checkPoint_path ...` (see "Pretraining on new datasets" above).
