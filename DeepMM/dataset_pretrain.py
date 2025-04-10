import os
import shutil
import random
import threading
import multiprocessing

import pysam
import numpy as np
from tqdm import tqdm


from .feature_encoder import *
from .tools import *

random.seed(42)

def balance_sample(folder_path):
    ne_file = []
    po_file = []

    for file in os.listdir(folder_path):
        label = int(file.split('_')[-1])
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

    for file in random_selection:
        shutil.rmtree(os.path.join(folder_path, file))

    po_file = [file for file in os.listdir(folder_path) if int(file.split('_')[-1]) == 1]
    ne_file = [file for file in os.listdir(folder_path) if int(file.split('_')[-1]) == 0]
    print(f"Final Positive samples: {len(po_file)}")
    print(f"Final Negative samples: {len(ne_file)}")


def get_views_feature(ori_point, comm_type, comm, contig_name, contig_seq, read_infos, is_avg, align_path, folder_path):
    random_number = random.random()
    #* Left shift
    pick_point = int(ori_point - (config['window_len'] / 2) * random_number)
    window_name = f"{comm_type}_comm_{comm}_{contig_name}_{pick_point}"
    window_feature, multiple_transloc, zero_fea, window_ts, window_bp = get_feature(point = pick_point, contig_seq=contig_seq, read_infos = read_infos, window_len = config['window_len'], is_avg=is_avg, align_path = align_path)
    if not multiple_transloc and not zero_fea:
        feature_file = os.path.join(folder_path, f'{window_name}_fw.npy')
        np.save(feature_file, window_feature)

    #* Right shift
    pick_point = int(ori_point + (config['window_len'] / 2) * random_number)
    window_name = f"{comm_type}_comm_{comm}_{contig_name}_{pick_point}"
    window_feature, multiple_transloc, zero_fea, window_ts, window_bp = get_feature(point = pick_point, contig_seq=contig_seq, read_infos = read_infos, window_len = config['window_len'], is_avg=is_avg, align_path = align_path)
    if not multiple_transloc and not zero_fea:
        feature_file = os.path.join(folder_path, f'{window_name}_bw.npy')
        np.save(feature_file, window_feature)

    #* Subsampling
    aligned_reads_count = len(read_infos)
    fold = (aligned_reads_count * 150) / len(contig_seq)
    pick_point = ori_point
    if fold > 20:
        sub_ratio = random.uniform(0.6, 1)
        sub_sample_reads = random.sample(read_infos, int(aligned_reads_count *  sub_ratio))
        window_name = f"{comm_type}_comm_{comm}_{contig_name}_{pick_point}"
        window_feature, multiple_transloc, zero_fea, window_ts, window_bp = get_feature(point = pick_point, contig_seq=contig_seq, read_infos = read_infos, window_len = config['window_len'], is_avg=is_avg, align_path = align_path)
        if not multiple_transloc and not zero_fea and all(window_feature[-1] > 0):
            feature_file = os.path.join(folder_path, f'{window_name}_sub.npy')
            np.save(feature_file, window_feature)
    else:
        window_name = f"{comm_type}_comm_{comm}_{contig_name}_{pick_point}"
        window_feature, multiple_transloc, zero_fea, window_ts, window_bp = get_feature(point = pick_point, contig_seq=contig_seq, read_infos = read_infos, window_len = config['window_len'], is_avg=is_avg, align_path = align_path)
        if not multiple_transloc and not zero_fea:
            feature_file = os.path.join(folder_path, f'{window_name}_sub.npy')
            np.save(feature_file, window_feature)

def check_and_remove_folder(folder_path):
    files = os.listdir(folder_path)
    if len(files) != config['n_views']:
        shutil.rmtree(folder_path)

def multi_view_collect(feature_folder, comm_type, comm, contig_name, label, align_path, pick_point, contig_seq, read_infos, is_avg):
    window_feature, multiple_transloc, zero_fea, window_ts, window_bp = get_feature(point = pick_point, contig_seq=contig_seq, read_infos = read_infos, window_len = config['window_len'], is_avg=is_avg, align_path = align_path)
    if not multiple_transloc and not zero_fea:
        window_name = f"{comm_type}_comm_{comm}_{contig_name}_{pick_point}"
        folder_path = f'{feature_folder}/{window_name}_{label}'
        os.makedirs(folder_path, exist_ok=True)
        feature_file = os.path.join(folder_path, f'{window_name}.npy')
        np.save(feature_file, window_feature)
        #* Reversed
        window_feature_reversed = []
        for i, row in enumerate(window_feature):
            if i != len(window_feature) - 2:
                window_feature_reversed.append(row[::-1])
            else:
                window_feature_reversed.append(row)
        window_feature_reversed = np.array(window_feature_reversed)
        feature_file = os.path.join(folder_path, f'{window_name}_re.npy')
        np.save(feature_file, window_feature_reversed)
        get_views_feature(pick_point, comm_type, comm, contig_name, contig_seq, read_infos, is_avg, align_path, folder_path)
        check_and_remove_folder(folder_path)

def get_positive_window(feature_folder, proper_bp_contig_names, contig_path, align_path, positive_contig, comm, comm_type):
    with pysam.FastaFile(contig_path) as contig_file:
        with pysam.AlignmentFile(align_path, 'rb') as align_file:
            for contig_name in tqdm(proper_bp_contig_names, desc = f'{comm_type}, Comm: {comm}, positive_cases'):
                contig_seq, read_infos = get_contig_infos(contig_name, contig_file, align_file)
                aligned_reads_count = len(read_infos)
                if aligned_reads_count >= 10:
                    contig_len = len(contig_seq)
                    skip_len = config['skip_perc'] * contig_len if config['skip_perc'] * contig_len < 1000 else 1000
                    is_avg = get_is_avg(read_infos, contig_seq)
                    for label_point in positive_contig[contig_name]['break_point']:
                        #* pick proper break point label
                        if skip_len + config['window_len'] / 2 <= label_point < contig_len - skip_len - config['window_len'] / 2:
                            random_number = random.random()
                            label = 1
                            #* Left shift
                            pick_point = int(label_point - (config['window_len'] / 2) * random_number)
                            multi_view_collect(feature_folder, comm_type, comm, contig_name, label, align_path, pick_point, contig_seq, read_infos, is_avg)

                            #* Middel
                            pick_point = int(label_point)
                            multi_view_collect(feature_folder, comm_type, comm, contig_name, label, align_path, pick_point, contig_seq, read_infos, is_avg)
                            
                            #* Right shift
                            pick_point = int(label_point + (config['window_len'] / 2) * random_number)
                            multi_view_collect(feature_folder, comm_type, comm, contig_name, label, align_path, pick_point, contig_seq, read_infos, is_avg)
                            

def get_negative_window(feature_folder, negative_contig_names, contig_path, align_path, comm, comm_type):
    with pysam.FastaFile(contig_path) as contig_file:
        with pysam.AlignmentFile(align_path, 'rb') as align_file:
            for contig_name in tqdm(negative_contig_names, desc = f'{comm_type}, Comm: {comm}, negative_cases'):
                contig_seq, read_infos = get_contig_infos(contig_name, contig_file, align_file)
                aligned_reads_count = len(read_infos)
                if aligned_reads_count >= 10:
                    contig_len = len(contig_seq)
                    skip_len = config['skip_perc'] * contig_len if config['skip_perc'] * contig_len < 1000 else 1000
                    is_avg = get_is_avg(read_infos, contig_seq)
                    #* Forward 
                    for pick_point in range(int(skip_len + config['window_len'] / 2), int(contig_len - skip_len - config['window_len'] / 2), config['window_len']):
                        label = 0
                        multi_view_collect(feature_folder, comm_type, comm, contig_name, label, align_path, pick_point, contig_seq, read_infos, is_avg)
                        
def get_pretrain_data(args):
    data_folder = args.data_folder
    feature_folder = args.feature_folder
    label = args.label
    assembly = args.assembly
    alignment = args.alignment
    os.makedirs(feature_folder, exist_ok=True)
    for comm_type in os.listdir(data_folder):
        community = os.listdir(os.path.join(data_folder, comm_type))
        community.sort(key=int)
        pool = multiprocessing.Pool(processes=30)
        for comm in community:
            label_path = os.path.join(data_folder, comm_type, comm, label)
            contig_path = os.path.join(data_folder, comm_type, comm, assembly)
            align_path = os.path.join(data_folder, comm_type, comm, alignment)
            negative_contig, positive_contig = extract_contig_info(label_path)

            # get contig length between min and max
            negative_contig_names = [contig_name for contig_name in negative_contig.keys() if config['min_contig_length'] <= negative_contig[contig_name]['contig_len'] < config['max_contig_length']]
            # get positve contig between min and max length
            positive_contig_names = [contig_name for contig_name in positive_contig.keys() if  config['min_contig_length'] <= positive_contig[contig_name]['contig_len'] < config['max_contig_length']]
            # divide into training set and evaluating set
            
            po_pre_train_end = int(0.8 * len(positive_contig_names))
            ne_pre_train_end = int(0.8 * len(negative_contig_names))

            positive_contig_names = positive_contig_names[:po_pre_train_end]
            negative_contig_names = negative_contig_names[:ne_pre_train_end]

            #! Bug test
            # get_positive_window(feature_folder, positive_contig_names, contig_path, align_path, positive_contig, comm, comm_type)
            # get_negative_window(feature_folder, negative_contig_names, contig_path, align_path, comm, comm_type)

            max_connections = threading.Semaphore(2)
            po_result = pool.apply_async(get_positive_window, (feature_folder, positive_contig_names, contig_path, align_path, positive_contig, comm, comm_type))
            ne_result = pool.apply_async(get_negative_window, (feature_folder, negative_contig_names, contig_path, align_path, comm, comm_type))

        pool.close()
        pool.join()
        balance_sample(feature_folder)
        print(len(os.listdir(feature_folder)))

