import re

import pysam
import numpy as np

from .tools import *


def get_window_features(point, contig_seq, read_infos, window_len, align_path):
    
    #* Initialize 
    b_window = int(point - (window_len / 2))
    e_window = b_window + window_len
    
    window_cl = np.zeros([window_len]).astype(np.float64)
    window_iv = np.zeros([window_len]).astype(np.float64)
    window_ts = np.zeros([window_len]).astype(np.float64)
    window_dp = np.zeros([window_len]).astype(np.float64)
    window_bp = np.zeros([window_len]).astype(np.float64)

    transloc_contig = [] 
    transloc_read_loc = []

    read_n = 0
    for read_info in read_infos:
        read_s = read_info.reference_start
        read_e = read_info.reference_end
        if read_s in range(b_window, e_window) or read_e in range(b_window, e_window):            
            align_s = 0 if read_s < b_window else read_s - b_window
            align_e = read_e - b_window if read_e < e_window else window_len
            read_n += 1
            window_dp[align_s: align_e] += 1

            #* Inverted orientation reads
            if read_info.mate_is_mapped and read_info.reference_name == read_info.next_reference_name:
                if ((read_info.is_reverse and read_info.mate_is_reverse) or (read_info.is_forward and read_info.mate_is_forward)): 
                    window_iv[align_s: align_e] += 1

            #* Translocated read pair
            elif read_info.mate_is_mapped and  read_info.reference_name != read_info.next_reference_name and read_info.mapping_quality == 60:
                with pysam.AlignmentFile(align_path, 'rb') as align_file:
                    try:
                        mate_info = align_file.mate(read_info)
                        mate_contig_name = mate_info.reference_name
                        mate_contig_length = align_file.get_reference_length(mate_contig_name)
                        if mate_info.mapping_quality >= config['mapping_quality'] and (read_info.next_reference_start in range(0, 300) or read_info.next_reference_start in range(mate_contig_length - 300, mate_contig_length)) and  mate_contig_length >= 3000:
                            transloc_contig.append(read_info.next_reference_name)
                            transloc_read_loc.append((align_s, align_e))
                            window_dp[align_s: align_e] -= 1
                    except Exception as e:
                        pass

            #* Clipped read
            if read_info.is_supplementary:
                if re.match('^([0-9]+H)', read_info.cigarstring):
                    h_match = re.search(r'\d+(?=H)', read_info.cigarstring)
                    clipped_len = int(h_match.group()) 
                    break_point = read_info.reference_start - b_window
                    if 0 <= break_point < window_len:
                        window_bp[break_point] += 1
                        window_cl[align_s: align_e] += clipped_len / read_info.query_length
                else:
                    break_point =  read_info.reference_end - b_window
                    if 0 <= break_point < window_len:
                        window_bp[break_point] += 1
                        window_cl[align_s: align_e] +=  (read_info.query_length - read_info.query_alignment_length) / read_info.query_length
                        

            if 0 < read_info.get_cigar_stats()[0][4]:
                numbers = re.findall(r'\d+(?=S)', read_info.cigarstring)
                clipped_len = [int(num) for num in numbers]
                if re.match('^([0-9]+S)',read_info.cigarstring):
                    break_point = read_info.reference_start - b_window
                    if 0 <= break_point < window_len:
                        window_bp[break_point] += 1
                        if len(clipped_len) == 1:
                            window_cl[align_s: align_e] += clipped_len[0] / read_info.query_length
                        elif len(clipped_len) == 2:
                            window_cl[align_s: align_e] += clipped_len[0] / read_info.query_length
                        
                if (read_info.cigarstring).endswith('S'): 
                    break_point =  read_info.reference_end - b_window
                    if 0 <= break_point < window_len:
                        window_bp[break_point] += 1
                        if len(clipped_len) == 1:
                            window_cl[align_s: align_e] += clipped_len[0] / read_info.query_length
                        elif len(clipped_len) == 2:
                            window_cl[align_s: align_e] += clipped_len[1] / read_info.query_length

    multiple_transloc = 0
    if len(set(transloc_contig)) > 3:
        multiple_transloc = 1
    else:
        for region in transloc_read_loc:
            align_s, align_e = region[0], region[1]
            window_dp[align_s: align_e] += 1
            window_ts[align_s: align_e] += 1

    window_cl[window_cl == 1] = 0
    window_ts[window_ts == 1] = 0
    window_iv[window_iv == 1] = 0
    return window_cl, window_iv, window_ts, window_dp, multiple_transloc, window_bp

def get_feature(point, contig_seq, read_infos, window_len, is_avg, align_path):
    point=int(point)
    window_cl, window_iv, window_ts, window_dp, multiple_transloc, window_bp = get_window_features(point = point, contig_seq = contig_seq, read_infos = read_infos, window_len = window_len, align_path = align_path)
    if all(window_cl == 0) and all(window_iv == 0) and all(window_ts == 0):
        return None, multiple_transloc, 1, window_ts, window_bp
    else:
        b_window = int(point - (window_len / 2))
        e_window = b_window + window_len
        window_is_avg = is_avg[b_window: e_window]
        window_feature = np.vstack((window_cl, window_iv, window_ts, window_is_avg, window_dp))
        return window_feature, multiple_transloc, 0, window_ts, window_bp
    
        
        