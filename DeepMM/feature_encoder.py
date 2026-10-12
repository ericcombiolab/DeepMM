import re

import pysam
import numpy as np

from .tools import *


def get_window_features(point, contig_seq, read_infos, window_len, align_path,
                        align_file=None, mate_cache=None, length_cache=None,
                        contig_name=None, allowed_query_names=None,
                        use_region_fetch=False):
    """Extract one window while preserving v1 feature semantics.

    In inference, ``use_region_fetch`` avoids scanning every read from the
    contig for every window.  Only indexed BAM records overlapping the current
    window are visited, then the same mapping filters and feature rules apply.
    Training callers keep the legacy read-list path by default.
    """
    #* Initialize
    b_window = int(point - (window_len / 2))
    e_window = b_window + window_len

    window_cl = np.zeros([window_len]).astype(np.float32)
    window_iv = np.zeros([window_len]).astype(np.float32)
    window_ts = np.zeros([window_len]).astype(np.float32)
    window_dp = np.zeros([window_len]).astype(np.float32)
    window_bp = np.zeros([window_len]).astype(np.float32)

    transloc_contig = []
    transloc_read_loc = []

    close_file = False
    if align_file is None:
        align_file = pysam.AlignmentFile(align_path, 'rb')
        close_file = True
    if mate_cache is None:
        mate_cache = {}
    if length_cache is None:
        length_cache = {}

    read_n = 0
    try:
        if use_region_fetch:
            if contig_name is None:
                raise ValueError("contig_name is required for BAM region fetch")
            # v1 counts a clipping breakpoint whose read end is exactly at
            # the left window boundary.  Fetch one base of halo so htslib
            # returns that record; the original endpoint predicate below
            # still decides whether it contributes.
            fetch_start = max(0, b_window - 1)
            fetch_end = min(len(contig_seq), e_window)
            candidate_reads = align_file.fetch(contig_name, fetch_start, fetch_end)
        else:
            candidate_reads = read_infos

        for read_info in candidate_reads:
            if use_region_fetch:
                if (
                    read_info.is_secondary
                    or read_info.is_duplicate
                    or not read_info.is_mapped
                    or read_info.mapping_quality < config['mapping_quality']
                    or not read_info.has_tag('AS')
                    or read_info.get_tag('AS') < config['alignment_score']
                ):
                    continue
                if (
                    allowed_query_names is not None
                    and read_info.query_name not in allowed_query_names
                ):
                    continue
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
                elif read_info.mate_is_mapped and read_info.reference_name != read_info.next_reference_name and read_info.mapping_quality == 60:
                    try:
                        cache_key = (read_info.query_name, read_info.is_read1)
                        if cache_key not in mate_cache:
                            try:
                                mate_cache[cache_key] = align_file.mate(read_info)
                            except Exception:
                                mate_cache[cache_key] = None
                        mate_info = mate_cache[cache_key]
                        if mate_info is not None:
                            mate_contig_name = mate_info.reference_name
                            if mate_contig_name not in length_cache:
                                length_cache[mate_contig_name] = align_file.get_reference_length(mate_contig_name)
                            mate_contig_length = length_cache[mate_contig_name]
                            if mate_info.mapping_quality >= config['mapping_quality'] and (read_info.next_reference_start in range(0, 300) or read_info.next_reference_start in range(mate_contig_length - 300, mate_contig_length)) and mate_contig_length >= 3000:
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
                        break_point = read_info.reference_end - b_window
                        if 0 <= break_point < window_len:
                            window_bp[break_point] += 1
                            window_cl[align_s: align_e] += (read_info.query_length - read_info.query_alignment_length) / read_info.query_length

                if 0 < read_info.get_cigar_stats()[0][4]:
                    numbers = re.findall(r'\d+(?=S)', read_info.cigarstring)
                    clipped_len = [int(num) for num in numbers]
                    if re.match('^([0-9]+S)', read_info.cigarstring):
                        break_point = read_info.reference_start - b_window
                        if 0 <= break_point < window_len:
                            window_bp[break_point] += 1
                            if len(clipped_len) == 1:
                                window_cl[align_s: align_e] += clipped_len[0] / read_info.query_length
                            elif len(clipped_len) == 2:
                                window_cl[align_s: align_e] += clipped_len[0] / read_info.query_length

                    if (read_info.cigarstring).endswith('S'):
                        break_point = read_info.reference_end - b_window
                        if 0 <= break_point < window_len:
                            window_bp[break_point] += 1
                            if len(clipped_len) == 1:
                                window_cl[align_s: align_e] += clipped_len[0] / read_info.query_length
                            elif len(clipped_len) == 2:
                                window_cl[align_s: align_e] += clipped_len[1] / read_info.query_length
    finally:
        if close_file:
            align_file.close()

    multiple_transloc = 0
    if len(set(transloc_contig)) > 3:
        multiple_transloc = 1
    else:
        for region in transloc_read_loc:
            align_s, align_e = region[0], region[1]
            window_dp[align_s: align_e] += 1
            window_ts[align_s: align_e] += 1

    window_ts[window_ts == 1] = 0
    window_iv[window_iv == 1] = 0
    return window_cl, window_iv, window_ts, window_dp, multiple_transloc, window_bp


def get_feature(point, contig_seq, read_infos, window_len, is_avg, align_path,
                align_file=None, mate_cache=None, length_cache=None,
                contig_name=None, allowed_query_names=None,
                use_region_fetch=False):
    point = int(point)
    window_cl, window_iv, window_ts, window_dp, multiple_transloc, window_bp = get_window_features(
        point=point, contig_seq=contig_seq, read_infos=read_infos, window_len=window_len,
        align_path=align_path, align_file=align_file, mate_cache=mate_cache,
        length_cache=length_cache, contig_name=contig_name,
        allowed_query_names=allowed_query_names,
        use_region_fetch=use_region_fetch
    )
    if all(window_cl == 0) and all(window_iv == 0) and all(window_ts == 0):
        return None, multiple_transloc, 1, window_ts, window_bp
    else:
        b_window = int(point - (window_len / 2))
        e_window = b_window + window_len
        window_is_avg = is_avg[b_window: e_window]
        window_feature = np.vstack((window_cl, window_iv, window_ts, window_is_avg, window_dp))
        return window_feature, multiple_transloc, 0, window_ts, window_bp
