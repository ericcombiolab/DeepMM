import os
import shutil
import random
import threading
import multiprocessing

import pysam
import numpy as np
from tqdm import tqdm

from .tools import *
from .feature_encoder import *

window_len = config['window_len']
min_contig_len = 3000
max_contig_len = float('inf')
skip_perc = config['skip_perc']
random.seed(42)


def balance_sample(folder_path):
    ne_file = []
    po_file = []

    for file in os.listdir(folder_path):
        label = int(file[:-4].split('_')[-1])
        if label == 1:
            po_file.append(file)
        elif label == 0:
            ne_file.append(file)
    
    po_num = len(po_file)
    ne_num = len(ne_file)

    print(f"Positive samples: {po_num}")
    print(f"Negative samples: {ne_num}")

    if po_num > ne_num:
        majority_files = po_file
        majority_label = 1
        num_to_remove = po_num - ne_num
    else:
        majority_files = ne_file
        majority_label = 0
        num_to_remove = ne_num - po_num

    print(f"Number of samples to remove: {num_to_remove}")

    random_selection = random.sample(majority_files, k=num_to_remove)

    for file in tqdm(random_selection, desc="Removing excess samples"):
        os.remove(os.path.join(folder_path, file))

    po_file = [file for file in os.listdir(folder_path) if int(file[:-4].split('_')[-1]) == 1]
    ne_file = [file for file in os.listdir(folder_path) if int(file[:-4].split('_')[-1]) == 0]
    print(f"Final Positive samples: {len(po_file)}")
    print(f"Final Negative samples: {len(ne_file)}")


def save_feature(window_name, window_feature, label, file_folder):
    file = f'{window_name}_{label}.npy'
    if file not in os.listdir(file_folder):
        np.save(os.path.join(file_folder, file), window_feature)

def get_positive_window(proper_bp_contig_names, contig_path, align_path, positive_contig, comm, file_folder):
    with pysam.FastaFile(contig_path) as contig_file:
        with pysam.AlignmentFile(align_path, 'rb') as align_file:
            for contig_name in tqdm(proper_bp_contig_names, desc = f'Comm: {comm}, positive_cases'):
                contig_seq, read_infos = get_contig_infos(contig_name, contig_file, align_file)
                if len(read_infos) >= 10:
                    contig_len = len(contig_seq)
                    skip_len = skip_perc * contig_len if skip_perc * contig_len < 1000 else 1000
                    is_avg = get_is_avg(read_infos, contig_seq)
                    for label_point in positive_contig[contig_name]['break_point']:
                        #* pick proper break point label
                        if skip_len + window_len / 2 <= label_point < contig_len - skip_len - window_len / 2:
                            #* Forward sliding
                            random_number = random.random()
                            point = int(label_point - (window_len / 2) * random_number)
                            window_feature, multiple_transloc, zero_fea, window_ts, window_bp = get_feature(point = point, contig_seq=contig_seq, read_infos = read_infos, window_len = config['window_len'], is_avg=is_avg, align_path = align_path)
                            if not multiple_transloc and not zero_fea:
                                window_name = f"comm_{comm}_{contig_name}_{point}"
                                label = 1
                                save_feature(window_name, window_feature, label, file_folder)

                            #* Midle one
                            point = int(label_point)
                            window_feature, multiple_transloc, zero_fea, window_ts, window_bp = get_feature(point = point, contig_seq=contig_seq, read_infos = read_infos, window_len = config['window_len'], is_avg=is_avg, align_path = align_path)
                            if not multiple_transloc and not zero_fea:
                                window_name = f"comm_{comm}_{contig_name}_{point}"
                                label = 1
                                save_feature(window_name, window_feature, label, file_folder)

                            #* Backward sliding
                            point = int(label_point + (window_len / 2) * random_number)
                            window_feature, multiple_transloc, zero_fea, window_ts, window_bp = get_feature(point = point, contig_seq=contig_seq, read_infos = read_infos, window_len = config['window_len'], is_avg=is_avg, align_path = align_path)
                            if not multiple_transloc and not zero_fea:
                                window_name = f"comm_{comm}_{contig_name}_{point}"
                                label = 1
                                save_feature(window_name, window_feature, label, file_folder)

def get_negative_window(negative_contig_names, contig_path, align_path, comm, file_folder):
    with pysam.FastaFile(contig_path) as contig_file:
        with pysam.AlignmentFile(align_path, 'rb') as align_file:
            for contig_name in tqdm(negative_contig_names, desc = f'Comm: {comm}, negative_cases'):
                contig_seq, read_infos = get_contig_infos(contig_name, contig_file, align_file)
                contig_len = len(contig_seq)
                if len(read_infos) >= 10:
                    skip_len = skip_perc * contig_len if skip_perc * contig_len < 1000 else 1000
                    is_avg = get_is_avg(read_infos, contig_seq)
                    #* Forward 
                    for point in range(int(skip_len + window_len / 2), int(contig_len - skip_len - window_len / 2), window_len):
                        window_feature, multiple_transloc, zero_fea, window_ts, window_bp = get_feature(point = point, contig_seq=contig_seq, read_infos = read_infos, window_len = config['window_len'], is_avg=is_avg, align_path = align_path)
                        if not multiple_transloc and not zero_fea:
                            window_name = f"comm_{comm}_{contig_name}_{point}"
                            label = 0
                            save_feature(window_name, window_feature, label, file_folder)


def get_training_data(args, status):
    data_folder = args.data_folder
    feature_folder = args.feature_folder
    label = args.label
    assembly = args.assembly
    alignment = args.alignment
    file_folder = f'{feature_folder}/{status}'
    os.makedirs(file_folder, exist_ok=True)
    community = os.listdir(data_folder)
    community.sort(key=int)
    pool = multiprocessing.Pool(processes=30)
    for comm in community:
        label_path = os.path.join(data_folder, comm, label)
        contig_path = os.path.join(data_folder, comm, assembly)
        align_path = os.path.join(data_folder, comm, alignment)
        negative_contig, positive_contig = extract_contig_info(label_path)

        negative_contig_names = [contig_name for contig_name in negative_contig.keys() if min_contig_len <= negative_contig[contig_name]['contig_len'] < max_contig_len]
        positive_contig_names = [contig_name for contig_name in positive_contig.keys() if  min_contig_len <= positive_contig[contig_name]['contig_len'] < max_contig_len]

        po_train_end = int(0.8 * len(positive_contig_names))
        ne_train_end = int(0.8 * len(negative_contig_names))

        if status == 'train':
            positive_contig_names = positive_contig_names[:po_train_end]
            negative_contig_names = negative_contig_names[:ne_train_end]
        elif status == 'eval':
            positive_contig_names = positive_contig_names[po_train_end:]
            negative_contig_names = negative_contig_names[ne_train_end:]

        max_connections = threading.Semaphore(2)
        po_result = pool.apply_async(get_positive_window, (positive_contig_names, contig_path, align_path, positive_contig, comm, file_folder))
        ne_result = pool.apply_async(get_negative_window, (negative_contig_names, contig_path, align_path, comm, file_folder))

    pool.close()
    pool.join()
    print('Data cleaning ... ')
    balance_sample(folder_path = f'{feature_folder}/{status}')

def get_fine_tune_dataset(args):
    if not args.eval_only:
        get_training_data(args, 'train')
        get_training_data(args, 'eval')
    else:
        get_training_data(args, 'eval')

