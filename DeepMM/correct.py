import os
import random
import subprocess
import multiprocessing
from functools import partial

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import pysam
from Bio import SeqIO
from tqdm import tqdm


from .tools import * 
from .model import *
from .feature_encoder import *

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def process_contig(contig_name, contig_file, align_file, model, device):
    bp_pred = []
    point_pred = []
    pred = []
    chimeric_pred = []
    
    contig_seq, read_infos = get_contig_infos(contig_name, contig_file, align_file)
    contig_len = len(contig_seq)
    if len(read_infos) >= 10:
        skip_len = config['skip_perc'] * contig_len if config['skip_perc'] * contig_len < 1000 else 1000
        is_avg, filtered_readInofs = get_is_avg(read_infos, contig_seq)
        for point in range(int(skip_len + 0.5 * config['window_len'] ), int(contig_len - skip_len - 0.5 * config['window_len']), config['window_len']):
            window_feature, multiple_transloc, zero_fea, window_ts, window_bp = get_feature(point = point, contig_seq=contig_seq, read_infos = filtered_readInofs, window_len = config['window_len'], is_avg=is_avg, align_file = align_file)
            if not multiple_transloc and not zero_fea:
                max_bp =  int(point - (config['window_len'] / 2)) + np.argmax(window_bp) if max(window_bp) > 0 else point
                feature_maps = df_difference_nor(window_feature)
                input_tensor = torch.Tensor(np.array(feature_maps)).to(device).unsqueeze(0)
                with torch.no_grad():
                    _, output = model(input_tensor)

                    max_value, max_index = torch.max(output, dim=1)
                    #* Class 1 -> misassembled
                    point_pred.append((point, output[0][1].item()))
                    pred.append(output[0][1].item())
                    if max_index.item() == 1 and max_value.item() >= 0.9:
                        bp_pred.append((max_bp, output[0][1].item()))
                        if not all(window_ts == 0):
                            chimeric_pred.append((max_bp, output[0][1].item()))

            else:
                point_pred.append((point, 0))
                pred.append(0)
    elif len(read_infos) <= 10:
        bp_pred = [(int(len(contig_seq) / 2), 1)]
        pred.append(1)

    return {
        'contig_name': contig_name,
        'bp_pred': bp_pred,
        'point_pred': point_pred,
        'max_prediction': max(pred),
        'chimeric_pred': chimeric_pred,
        'contig_len': contig_len,
        'window_prediction': pred
    }

def correct(args):
    
    threads = args.threads
    device = args.gpu_device
    fine_tune = args.fine_tune
    folder_path = args.folder_path
    model_weight_path = args.model_weight_path
    file_name = args.file_name
    contig_file_name = args.contig_file_name
    bam_file_name = args.bam_file_name
    correct_all = args.correct_all
    score_cut = args.threshold
    

    multiprocessing.set_start_method('spawn')
    contig_path = os.path.join(folder_path, contig_file_name)
    align_path = os.path.join(folder_path, bam_file_name)

    contig_fai_path = f'{contig_path}.fai'
    align_fai_path = f'{align_path}.bai'

    touch_cmd = [
        'touch', contig_fai_path
    ]
    subprocess.run(touch_cmd)
    touch_cmd = [
        'touch', align_fai_path
    ]
    subprocess.run(touch_cmd)

    output_folder_path = f'{folder_path}/deepmm_output'
    os.makedirs(output_folder_path, exist_ok=True)
    
    model = LinearEvaluation(ResNetSimCLR(base_model='resnet50', out_dim=128))

    if fine_tune:
        print("Using Fine-tune model")
        model = FinetuneModel(model)
    else:
        print("Using Default model")

    model.load_state_dict(torch.load(model_weight_path, map_location = device, weights_only=True))
    model = model.to(device)
    model.eval()   

    contig_test = {}
    total_contig_name = []
    with pysam.FastaFile(contig_path) as contig_file:
        with pysam.AlignmentFile(align_path) as align_file:
            all_contig_names = contig_file.references
            for contig_name in all_contig_names:
                contig_seq = contig_file.fetch(contig_name)
                if config['min_contig_length'] <= len(contig_seq) < config['max_contig_length']:
                    total_contig_name.append(contig_name)

            process_contig_partial = partial(process_contig, contig_file=contig_file, align_file=align_file, model = model, device = device)

            with multiprocessing.Pool(processes=threads) as pool:
                results = list(tqdm(pool.imap(process_contig_partial, total_contig_name), total=len(total_contig_name), desc='Predicting ...'))

            contig_test = {result['contig_name']: {'bp_pred': result['bp_pred'], 'point_pred': result['point_pred'],
                                                    'max_prediction': result['max_prediction'], 'contig_len': result['contig_len'],
                                                    'chimeric_pred': result['chimeric_pred'], 'window_prediction': result['window_prediction']
                                                    } for result in results}

    breakcontigs = []
    for contig_name in contig_test.keys():
        if len(contig_test[contig_name]['bp_pred']) > 0:
            breakcontigs.append(contig_name)
            
    with open(f'{output_folder_path}/{file_name}', 'w') as f:
        f.write('Assembly\tBreakPoint\tPrediction\tLength\tChimeric_BreakPoint\tChimeric_Prediction\n')
        for contig_name in total_contig_name:
            if contig_name in breakcontigs:
                sorted_data = sorted(contig_test[contig_name]['bp_pred'], key=lambda x: x[1])
                bp = sorted_data[-1][0]
                bp_prediction = sorted_data[-1][1]
                if len(contig_test[contig_name]['chimeric_pred']) > 0:
                    sorted_data = sorted(contig_test[contig_name]['chimeric_pred'], key=lambda x: x[1])
                    chi_point = sorted_data[-1][0]
                    chi_prediction = sorted_data[-1][1]
                    f.write(contig_name + '\t' + str(bp) + '\t' + str(bp_prediction) + '\t' + str(contig_test[contig_name]['contig_len']) + '\t' + str(chi_point) + '\t' + str(chi_prediction) + '\n')
                else:
                    f.write(contig_name + '\t' + str(bp) + '\t' + str(bp_prediction) + '\t' + str(contig_test[contig_name]['contig_len']) + '\t' + '-1' + '\t' + '0' + '\n')
            else:
                f.write(contig_name + '\t' + '-1' + '\t' + str(contig_test[contig_name]['max_prediction']) + '\t' + str(contig_test[contig_name]['contig_len']) + '\t' + '-1' + '\t' + '0' + '\n')

    corrected_contig_file = os.path.join(output_folder_path, "corrected_contigs.fa")
    original_file = f'{folder_path}/{contig_file_name}'
    input_file = SeqIO.parse(original_file, "fasta")
    df = pd.read_csv(f'{output_folder_path}/{file_name}', sep='\t')
    
    if not correct_all:
        print('Correcting chimeric misassembly')
        breakcontigs = df.loc[df['Chimeric_Prediction'] > score_cut,]
        breakcontigs = breakcontigs.loc[breakcontigs['Chimeric_BreakPoint'] > config['min_split_length'], ]
        breakcontigs = breakcontigs.loc[(breakcontigs['Length'] - breakcontigs['Chimeric_BreakPoint']) > config['min_split_length'], ]
        breakcontigs = list(np.unique(breakcontigs['Assembly']))
    else:
        print('Correcting all misassemblies')
        breakcontigs = df.loc[df['Prediction'] > score_cut,]
        breakcontigs = breakcontigs.loc[breakcontigs['BreakPoint'] > config['min_split_length'], ]
        breakcontigs = breakcontigs.loc[(breakcontigs['Length'] - breakcontigs['BreakPoint']) > config['min_split_length'], ]
        breakcontigs = list(np.unique(breakcontigs['Assembly']))
    
    with open(corrected_contig_file, "w") as corrected_file:
        for record in tqdm(input_file, desc = f'Correting ... '):
            if record.id in breakcontigs:
                if not correct_all:
                    bp = int(df.loc[df['Assembly'] == record.id, 'Chimeric_BreakPoint'].values[0])
                else:
                    bp = int(df.loc[df['Assembly'] == record.id, 'BreakPoint'].values[0])
                corrected_file.write(">" + record.id + "_1\n")
                corrected_file.write(str(record.seq[:bp]) + "\n")
                corrected_file.write(">" + record.id + "_2\n")
                corrected_file.write(str(record.seq[bp:]) + "\n")
            else:
                corrected_file.write(">" + record.id + "\n")
                corrected_file.write(str(record.seq) + "\n")
    print(f"A total of " + str(len(breakcontigs)) + " misassembled contigs are corrected")
