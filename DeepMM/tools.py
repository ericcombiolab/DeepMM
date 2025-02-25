import os
import glob
import statistics
import numpy as np
from scipy.ndimage import gaussian_filter

import torch
from torch.utils.data import Dataset

config = {
    'window_len':384,
    'min_contig_length':5000,
    'max_contig_length': float('inf'),
    'skip_perc': 0.10,
    'feature_channels': 6,
    'mapping_quality': 40,
    'alignment_score': 100,
    'threshold': 0.9,
    'n_views': 5,
    'min_split_length':1000
}

def get_contig_infos(contig_name, contig_file, align_file):
    contig_seq = contig_file.fetch(contig_name)
    aligned_reads = align_file.fetch(contig_name)
    read_infos = [
                    read_info for read_info in aligned_reads
                    if not read_info.is_secondary
                    and not read_info.is_duplicate
                    and read_info.is_mapped
                    and read_info.mapping_quality >= config['mapping_quality']
                    and read_info.get_tag('AS') >= config['alignment_score']
                 ]
    return contig_seq, read_infos

def metrics_cal(TP, TN, FN, FP):
    print(f'total: {TP + FP + FN + TN}')
    print(f'TP:{TP}\tTN:{TN}\tFN:{FN}\tFP:{FP}')
    accuracy = round((TP + TN) / (TP + FP + FN + TN), 3)
    print(f'accuracy: {accuracy}')
    precision = round(TP / (TP + FP), 3)
    print(f'precision: {precision}')
    recall = round(TP / (TP + FN), 3)
    print(f'recall: {recall}')
    f1 = round(2*precision*recall / (precision + recall), 3)
    print(f'f1 score: {f1}')

def get_is_avg(read_infos, contig_seq):
    contig_is = {i: [] for i in range(len(contig_seq))}
    pool={}
    for read_info in read_infos:
        if  read_info.mate_is_mapped and read_info.reference_name == read_info.next_reference_name:
            #* Insert size
            if read_info.query_name in pool and pool[read_info.query_name]['contig_name'] == read_info.reference_name:
                mate_read = pool[read_info.query_name]
                pe_s = min(read_info.reference_start, mate_read['read_s']) 
                pe_e = max(read_info.reference_end, mate_read['read_e'])
                size = pe_e - pe_s
                mid_s = max(read_info.reference_start, mate_read['read_s'])
                mid_e = min(read_info.reference_end, mate_read['read_e'])
                if mid_s < mid_e:
                    for i in range(read_info.reference_start, read_info.reference_end):
                        contig_is[i].append(size)
                    for i in range(mate_read['read_s'],  mate_read['read_e']):
                        contig_is[i].append(size)
                else:
                    for i in range(pe_s, pe_e):
                        contig_is[i].append(size)
            else:
                pool[read_info.query_name] = {
                    'contig_name': read_info.reference_name,
                    'read_s': read_info.reference_start,
                    'read_e': read_info.reference_end
                }
    
    is_avg = np.array([statistics.mean(values) if len(values) > 0 else 0 for values in contig_is.values()])
    return is_avg

def df_difference_nor(features_npy):
    dp_non_zero_mask = features_npy[4] != 0
    is_non_zero_mask = features_npy[3] != 0
    for i in range(3):
        features_npy[i][dp_non_zero_mask] = np.divide(features_npy[i][dp_non_zero_mask], features_npy[4][dp_non_zero_mask])
    
    features_npy[4][dp_non_zero_mask] = np.log(features_npy[4][dp_non_zero_mask])
    dp_diff_feature_map = np.abs(np.subtract.outer(features_npy[4], features_npy[4]))
    
    #* Insert size refilled
    diff_mask = ~np.logical_and(is_non_zero_mask, dp_non_zero_mask)
    for i in range(len(features_npy[3])):
        if diff_mask[i]:
            j = i - 1
            while j >= 0 and features_npy[3][j] == 0:
                j -= 1
            if j >= 0:
                features_npy[3][i] = features_npy[3][j]
            else:
                j = i + 1
                while j < len(features_npy[3]) and features_npy[3][j] == 0:
                    j += 1
                if j < len(features_npy[3]):
                    features_npy[3][i] = features_npy[3][j]
                else:
                    features_npy[3][i] = 0

    #* Insert size standardization
    is_non_zero_mask = features_npy[3] != 0
    non_zero_values = features_npy[3][is_non_zero_mask]
    mean = np.mean(non_zero_values)
    std_dev = np.std(non_zero_values)
    
    if std_dev == 0:
        features_npy[3] = features_npy[3] * 0
    else:
        features_npy[3][is_non_zero_mask] = (features_npy[3][is_non_zero_mask] - mean) / std_dev

    #* Gaussian filter
    for i in range(len(features_npy)):
        features_npy[i] = gaussian_filter(features_npy[i], sigma=3)
    dp_diff_feature_map = gaussian_filter(dp_diff_feature_map, sigma=3)
    feature_maps = [np.diag(features_npy[i]) for i in range(len(features_npy))]
    feature_maps.append(dp_diff_feature_map)
    return feature_maps

class Pretrain_FeatureDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.feature_file_path = path
        self.feature_files = os.listdir(self.feature_file_path)
        self.length = len(self.feature_files)

    def __getitem__(self, index):

        label = int(self.feature_files[index].split('_')[-1])
        window_name = '_'.join(self.feature_files[index].split('_')[:-1])

        features_npy = np.load(os.path.join(self.feature_file_path, self.feature_files[index], f'{window_name}.npy'), allow_pickle=True)
        feature = torch.Tensor(np.array(df_difference_nor(features_npy)))

        file_pattern = os.path.join(self.feature_file_path, self.feature_files[index], '*_fw.npy')
        file = glob.glob(file_pattern)[0]
        fw_features_npy = np.load(file, allow_pickle=True)
        fw_feature = torch.Tensor(np.array(df_difference_nor(fw_features_npy)))

        file_pattern = os.path.join(self.feature_file_path, self.feature_files[index], '*_bw.npy')
        file = glob.glob(file_pattern)[0]
        bw_features_npy = np.load(file, allow_pickle=True)
        bw_feature = torch.Tensor(np.array(df_difference_nor(bw_features_npy)))

        file_pattern = os.path.join(self.feature_file_path, self.feature_files[index], '*_re.npy')
        file = glob.glob(file_pattern)[0]
        re_feature_npy = np.load(file, allow_pickle=True)
        re_feature = torch.Tensor(np.array(df_difference_nor(re_feature_npy)))

        file_pattern = os.path.join(self.feature_file_path, self.feature_files[index], '*_sub.npy')
        file = glob.glob(file_pattern)[0]
        sub_feature_npy = np.load(file, allow_pickle=True)
        sub_feature = torch.Tensor(np.array(df_difference_nor(sub_feature_npy)))

        if label == 1:
            return (feature, fw_feature, bw_feature, sub_feature, re_feature), torch.Tensor([0, 1])
        elif label == 0:
            return (feature, fw_feature, bw_feature, sub_feature, re_feature), torch.Tensor([1, 0])

    def __len__(self):
        return self.length

class EvalDataset(Dataset):
    def __init__(self, path, status):
        super().__init__()
        self.status = status
        self.feature_file_path = os.path.join(path, status)
        self.feature_files = os.listdir(self.feature_file_path)
        self.length = len(self.feature_files)
        self.img_channel = config['feature_channels']

    def __getitem__(self, index):
        window_name = self.feature_files[index][:-6]
        features_npy = np.load(os.path.join(self.feature_file_path, self.feature_files[index]), allow_pickle=True)
        feature = torch.Tensor(np.array(df_difference_nor(features_npy)))
        label = self.feature_files[index][:-4].split('_')[-1]
        if label == '1':
            return window_name, feature, torch.Tensor([0, 1])
        elif label == '0':
            return window_name, feature, torch.Tensor([1, 0])
        
    def __len__(self):
        return self.length
    
