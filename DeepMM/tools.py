import os
import glob
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
    'n_views': 5,
    'min_split_length':1000,
    'window_threads': 8
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
    """Insert-size profile using O(1) interval updates per read pair.

    Difference arrays preserve the v1 interval semantics while avoiding tens
    of thousands of repeated NumPy slice updates on long/high-depth contigs.
    """
    L = len(contig_seq)
    sums_diff = np.zeros(L + 1, dtype=np.float64)
    counts_diff = np.zeros(L + 1, dtype=np.int64)
    pool = {}
    filtered_query = set()

    def add_interval(start, end, size):
        start = max(0, min(int(start), L))
        end = max(0, min(int(end), L))
        if start >= end:
            return
        sums_diff[start] += size
        sums_diff[end] -= size
        counts_diff[start] += 1
        counts_diff[end] -= 1

    for read_info in read_infos:
        if read_info.mate_is_mapped and read_info.reference_name == read_info.next_reference_name:
            #* Insert size
            if read_info.query_name in pool and pool[read_info.query_name]['contig_name'] == read_info.reference_name:
                mate_read = pool[read_info.query_name]
                if read_info.reference_start == mate_read['read_s']:
                    filtered_query.add(read_info.query_name)
                else:
                    pe_s = min(read_info.reference_start, mate_read['read_s'])
                    pe_e = max(read_info.reference_end, mate_read['read_e'])
                    size = pe_e - pe_s
                    mid_s = max(read_info.reference_start, mate_read['read_s'])
                    mid_e = min(read_info.reference_end, mate_read['read_e'])
                    if mid_s < mid_e:
                        add_interval(
                            read_info.reference_start,
                            read_info.reference_end,
                            size,
                        )
                        add_interval(
                            mate_read['read_s'],
                            mate_read['read_e'],
                            size,
                        )
                    else:
                        add_interval(pe_s, pe_e, size)
            else:
                pool[read_info.query_name] = {
                    'contig_name': read_info.reference_name,
                    'read_s': read_info.reference_start,
                    'read_e': read_info.reference_end
                }
    sums = np.cumsum(sums_diff[:-1])
    counts = np.cumsum(counts_diff[:-1])
    filtered_readInofs = [
        read_info for read_info in read_infos
        if read_info.query_name not in filtered_query
    ]
    is_avg = np.zeros(L, dtype=np.float64)
    try:
        mask = counts > 0
        is_avg[mask] = sums[mask] / counts[mask]
    except Exception as e:
        pass
    return is_avg, filtered_readInofs

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
    try:
        mean = np.mean(non_zero_values)
        std_dev = np.std(non_zero_values)
    except Exception as e:
        pass
    
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

def extract_contig_info(file_path, strict_negative=True):
    """Extract binary contig labels from MetaQUAST's irregular alignment TSV.

    Positive parsing intentionally preserves the project-specific legacy
    behavior.  By default, negatives are restricted to high-confidence clean
    contigs: one unambiguous best-group alignment covering the full contig at
    100% identity, with no local misassembly marker.  Set
    ``strict_negative=False`` to reproduce the previous IDY>=98 behavior.
    """
    negative_contig = {}
    positive_contig = {}
    mis_type = ['relocation', 'inversion', 'interspecies', 'translocation']
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            content = line.strip('\n').split('\t')
            if 'CONTIG' in content and ('correct' in content or 'misassembled' in content):
                contig_name = content[1]
                contig_len = int(content[2])
                mapping_percentage = float(lines[i-1].strip('\n').split('\t')[6])
                legacy_clean = (
                    (content[-1] == 'correct' or content[-1] == 'correct_unaligned')
                    and 'local misassembly' not in lines[i-2]
                    and mapping_percentage >= 98
                    and 'CONTIG' in lines[i-2]
                )
                strict_clean = False
                if legacy_clean and content[-1] == 'correct':
                    alignment = lines[i-1].strip('\n').split('\t')
                    try:
                        s2 = int(alignment[2])
                        e2 = int(alignment[3])
                        identity = float(alignment[6])
                        ambiguous = alignment[7].strip().lower() == 'true'
                        best_group = alignment[-1].strip().lower() == 'true'
                        full_length = min(s2, e2) == 1 and max(s2, e2) == contig_len
                        strict_clean = (
                            np.isclose(identity, 100.0)
                            and full_length
                            and not ambiguous
                            and best_group
                        )
                    except (IndexError, ValueError):
                        strict_clean = False
                if (strict_clean if strict_negative else legacy_clean):
                    negative_contig[contig_name] = {'contig_len': contig_len}
                elif content[-1] == 'misassembled' or 'local misassembly' in lines[i-2]:
                    point = []
                    k = 1
                    while True:
                        if 'CONTIG' in lines[i - k]:
                            break
                        else:
                            if contig_name in lines[i - k]:
                                if lines[i-k].strip('\n').split('\t')[-1] == 'True':
                                    point.extend([int(lines[i-k].strip('\n').split('\t')[2]), int(lines[i-k].strip('\n').split('\t')[3])])
                        k += 1
                    point = [x for x in point if x != 1 and x != contig_len]
                    point.sort()
                    arr = np.array(point)
                    diff = np.diff(arr)
                    mask = np.concatenate(([True], np.abs(diff) != 1))
                    result = list(set(arr[mask]))
                    result.sort()
                    if content[-1] == 'misassembled':
                        k = 2
                        while True:
                            mis_info = lines[i - k].split()[0].replace(',', '')
                            if mis_info in mis_type:
                                positive_contig[contig_name] = {'break_point': result, 'contig_len': contig_len, 'mis_type': mis_info}
                                break
                            k += 2
    return negative_contig, positive_contig

class Pretrain_FeatureDataset(Dataset):
    def __init__(self, path, preload=True, npz_dir=None):
        super().__init__()
        self.feature_file_path = path
        self._preload = preload
        self._npz_dir = npz_dir
        # Support both flat layout (window dirs directly under path) and
        # nested unit layout (window dirs under unit subdirectories).
        if any(os.path.isdir(os.path.join(path, d)) and
               os.path.basename(d).split('_')[-1] in ('0', '1') for d in os.listdir(path)):
            self.feature_files = [d for d in os.listdir(path)
                                  if os.path.isdir(os.path.join(path, d))]
        else:
            self.feature_files = []
            for unit in os.listdir(path):
                unit_path = os.path.join(path, unit)
                if not os.path.isdir(unit_path):
                    continue
                for d in os.listdir(unit_path):
                    if os.path.isdir(os.path.join(unit_path, d)) and \
                       os.path.basename(d).split('_')[-1] in ('0', '1'):
                        self.feature_files.append(os.path.join(unit, d))
        self.length = len(self.feature_files)
        self._cache = {}
        if self._preload:
            import time as _t
            t0 = _t.time()
            self._mem = {}
            if self._npz_dir:
                # Load from pre-packed per-unit npz files: fast, few big files.
                for unit in os.listdir(self._npz_dir):
                    if not unit.endswith('.npz'):
                        continue
                    zp = os.path.join(self._npz_dir, unit)
                    z = np.load(zp)
                    for key in z.files:
                        wdir, vname = key.rsplit('__', 1)
                        full = os.path.join(unit[:-4], wdir)
                        self._mem.setdefault(full, {})[vname] = z[key]
                    z.close()
            else:
                from concurrent.futures import ThreadPoolExecutor
                from tqdm import tqdm as _tqdm

                def _load_one(wdir):
                    win = os.path.join(self.feature_file_path, wdir)
                    base = os.path.basename(wdir)
                    wname = '_'.join(base.split('_')[:-1])
                    return wdir, {
                        wname: np.load(os.path.join(win, f'{wname}.npy'), allow_pickle=True),
                        wname + '_fw': np.load(os.path.join(win, f'{wname}_fw.npy'), allow_pickle=True),
                        wname + '_bw': np.load(os.path.join(win, f'{wname}_bw.npy'), allow_pickle=True),
                        wname + '_re': np.load(os.path.join(win, f'{wname}_re.npy'), allow_pickle=True),
                        wname + '_sub': np.load(os.path.join(win, f'{wname}_sub.npy'), allow_pickle=True),
                    }

                n_threads = min(32, os.cpu_count() or 1)
                with ThreadPoolExecutor(max_workers=n_threads) as _ex:
                    for _wdir, _data in _tqdm(
                        _ex.map(_load_one, self.feature_files),
                        total=self.length,
                        desc='Preloading features',
                    ):
                        self._mem[_wdir] = _data
            print(f'[Pretrain_FeatureDataset] preloaded {self.length} windows in {_t.time()-t0:.1f}s', flush=True)

    def __getitem__(self, index):

        window_dir = self.feature_files[index]
        base = os.path.basename(window_dir)
        label = int(base.split('_')[-1])
        window_name = '_'.join(base.split('_')[:-1])
        win_path = os.path.join(self.feature_file_path, window_dir)

        if self._preload:
            mem = self._mem[window_dir]
            # copy() so df_difference_nor (which mutates in place) does not
            # trigger copy-on-write page duplication across DataLoader workers.
            features_npy = mem[window_name].copy()
            fw_features_npy = mem[window_name + '_fw'].copy()
            bw_features_npy = mem[window_name + '_bw'].copy()
            re_feature_npy = mem[window_name + '_re'].copy()
            sub_feature_npy = mem[window_name + '_sub'].copy()
        else:
            features_npy = np.load(os.path.join(win_path, f'{window_name}.npy'), allow_pickle=True)

            file_pattern = os.path.join(win_path, '*_fw.npy')
            file = glob.glob(file_pattern)[0]
            fw_features_npy = np.load(file, allow_pickle=True)

            file_pattern = os.path.join(win_path, '*_bw.npy')
            file = glob.glob(file_pattern)[0]
            bw_features_npy = np.load(file, allow_pickle=True)

            file_pattern = os.path.join(win_path, '*_re.npy')
            file = glob.glob(file_pattern)[0]
            re_feature_npy = np.load(file, allow_pickle=True)

            file_pattern = os.path.join(win_path, '*_sub.npy')
            file = glob.glob(file_pattern)[0]
            sub_feature_npy = np.load(file, allow_pickle=True)

        feature = torch.Tensor(np.array(df_difference_nor(features_npy)))
        fw_feature = torch.Tensor(np.array(df_difference_nor(fw_features_npy)))
        bw_feature = torch.Tensor(np.array(df_difference_nor(bw_features_npy)))
        re_feature = torch.Tensor(np.array(df_difference_nor(re_feature_npy)))
        sub_feature = torch.Tensor(np.array(df_difference_nor(sub_feature_npy)))

        if label == 1:
            result = (feature, fw_feature, bw_feature, sub_feature, re_feature), torch.Tensor([0, 1])
        elif label == 0:
            result = (feature, fw_feature, bw_feature, sub_feature, re_feature), torch.Tensor([1, 0])
        return result

    def __len__(self):
        return self.length

class EvalDataset(Dataset):
    def __init__(self, path, status=None):
        super().__init__()
        self.img_channel = config['feature_channels']
        # 定位特征根目录：优先 path/status，否则回退到 path 本身。
        # 兼容两种调用：
        #   - finetune:  EvalDataset(root, 'train'/'eval') -> root/train, root/eval
        #   - pretrain:  EvalDataset(val_dir, 'eval') 但数据直接在 val_dir 下（无 eval 子目录）
        if status:
            cand = os.path.join(path, status)
            root = cand if os.path.isdir(cand) else path
        else:
            root = path
        if not os.path.isdir(root):
            raise FileNotFoundError(f'EvalDataset: directory not found: {root}')
        self.feature_file_path = root
        self._index = self._build_index(root)
        self.length = len(self._index)

    def _build_index(self, root):
        """返回 [(npy_path, label, window_name), ...]，兼容 flat 与 nested 布局。

        flat:   root/*.npy，文件名末尾 _0/_1 是标签
        nested: root/unit/窗口目录/{wname}.npy，窗口目录名末尾 _0/_1 是标签
        """
        items = []
        entries = sorted(os.listdir(root))
        has_npy = any(e.endswith('.npy') for e in entries)
        if has_npy:
            for fn in entries:
                if not fn.endswith('.npy'):
                    continue
                label = fn[:-4].split('_')[-1]
                if label not in ('0', '1'):
                    continue
                window_name = fn[:-6]
                items.append((os.path.join(root, fn), label, window_name))
        else:
            for unit in entries:
                unit_path = os.path.join(root, unit)
                if not os.path.isdir(unit_path):
                    continue
                for wdir in sorted(os.listdir(unit_path)):
                    wpath = os.path.join(unit_path, wdir)
                    if not os.path.isdir(wpath):
                        continue
                    label = wdir.split('_')[-1]
                    if label not in ('0', '1'):
                        continue
                    window_name = wdir.rsplit('_', 1)[0]
                    base_npy = os.path.join(wpath, window_name + '.npy')
                    if not os.path.isfile(base_npy):
                        continue
                    items.append((base_npy, label, window_name))
        return items

    def __getitem__(self, index):
        npy_path, label, window_name = self._index[index]
        features_npy = np.load(npy_path, allow_pickle=True)
        feature = torch.Tensor(np.array(df_difference_nor(features_npy)))
        if label == '1':
            return window_name, feature, torch.Tensor([0, 1])
        elif label == '0':
            return window_name, feature, torch.Tensor([1, 0])

    def __len__(self):
        return self.length
    
