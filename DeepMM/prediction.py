import os
import gc
import json
import random
import subprocess
import multiprocessing
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import torch

import pysam
import numpy as np
from tqdm import tqdm

from .tools import *
from .model import *
from .feature_encoder import *
from .escore import run as run_escore, describe as describe_escore, E_P05

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)

# Keep deterministic for parity with pretrained weights / reproducible inference
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def ensure_indexes(contig_path, align_path):
    """Create FASTA/BAM indexes if missing (replaces blind touch)."""
    fai = f'{contig_path}.fai'
    bai = f'{align_path}.bai'
    if (not os.path.isfile(fai)) or os.path.getsize(fai) == 0:
        subprocess.run(['samtools', 'faidx', contig_path], check=True)
    if (not os.path.isfile(bai)) or os.path.getsize(bai) == 0:
        subprocess.run(['samtools', 'index', align_path], check=True)


def extract_windows_for_contig(contig_name, contig_path, align_path):
    """CPU-only: extract compact raw window vectors (no expanded 2D maps).

    Expanding every window to 6x384x384 inside a worker made one long contig
    consume gigabytes and forced that payload through a multiprocessing pipe.
    The exact same ``df_difference_nor`` transform is now applied just-in-time
    for each GPU batch.
    """
    with pysam.FastaFile(contig_path) as contig_file:
        with pysam.AlignmentFile(align_path, 'rb') as align_file:
            contig_seq, read_infos = get_contig_infos(contig_name, contig_file, align_file)

            contig_len = len(contig_seq)
            windows = []

            if len(read_infos) < 10:
                return {
                    'contig_name': contig_name,
                    'contig_len': contig_len,
                    'low_coverage': True,
                    'windows': [],
                }

            skip_len = min(config['skip_perc'] * contig_len, 1000)
            is_avg, filtered_readInofs = get_is_avg(read_infos, contig_seq)
            allowed_query_names = {
                read_info.query_name for read_info in filtered_readInofs
            }
            window_len = config['window_len']
            window_len_half = 0.5 * window_len
            mate_cache = {}
            length_cache = {}

            start_point = int(skip_len + window_len_half)
            end_point = int(contig_len - skip_len - window_len_half)

            for point in range(start_point, end_point, window_len):
                try:
                    window_feature, multiple_transloc, zero_fea, window_ts, window_bp = get_feature(
                        point=point,
                        contig_seq=contig_seq,
                        read_infos=filtered_readInofs,
                        window_len=window_len,
                        is_avg=is_avg,
                        align_path=align_path,
                        align_file=align_file,
                        mate_cache=mate_cache,
                        length_cache=length_cache,
                        contig_name=contig_name,
                        allowed_query_names=allowed_query_names,
                        use_region_fetch=True,
                    )
                    if not multiple_transloc and not zero_fea:
                        max_bp = int(point - window_len_half) + np.argmax(window_bp) if np.max(window_bp) > 0 else point
                        windows.append({
                            'point': point,
                            'max_bp': max_bp,
                            'raw_features': np.asarray(
                                window_feature, dtype=np.float32
                            ),
                            'window_ts': window_ts,
                            'skip_score': False,
                        })
                    else:
                        windows.append({
                            'point': point,
                            'max_bp': point,
                            'raw_features': None,
                            'window_ts': window_ts,
                            'skip_score': True,
                        })
                except Exception as e:
                    print(f"Error processing window at point {point} on {contig_name}: {e}")
                    windows.append({
                        'point': point,
                        'max_bp': point,
                        'raw_features': None,
                        'window_ts': None,
                        'skip_score': True,
                    })

            return {
                'contig_name': contig_name,
                'contig_len': contig_len,
                'low_coverage': False,
                'windows': windows,
            }


def _count_infer_windows(result):
    if result.get('low_coverage'):
        return 0
    n = 0
    for w in result['windows']:
        if (not w['skip_score']) and w['raw_features'] is not None:
            n += 1
    return n


def run_batched_inference_chunk(model, device, contig_results, score_cut, batch_size):
    """GPU batch inference for one contig chunk; drop feature tensors after scoring.

    Same per-window math / aggregation rules as previous Opt1, but only keeps a
    compact prediction dict so peak RSS stays proportional to chunk size.
    """
    contig_test = {}

    jobs = []
    for ci, result in enumerate(contig_results):
        if result.get('low_coverage'):
            continue
        for wi, w in enumerate(result['windows']):
            if (not w['skip_score']) and w['raw_features'] is not None:
                jobs.append((ci, wi))

    scores = {}
    n_jobs = len(jobs)
    if n_jobs > 0:
        def expand(raw):
            fm = df_difference_nor(np.array(raw, copy=True))
            return torch.as_tensor(np.asarray(fm, dtype=np.float32))

        workers = max(1, int(config.get('window_threads', 1)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for start in range(0, n_jobs, batch_size):
                batch_jobs = jobs[start:start + batch_size]
                raws = [
                    contig_results[ci]['windows'][wi]['raw_features']
                    for ci, wi in batch_jobs
                ]
                if workers > 1:
                    tensors = list(executor.map(expand, raws))
                else:
                    tensors = [expand(raw) for raw in raws]
                for ci, wi in batch_jobs:
                    # Release compact raw feature after image construction.
                    contig_results[ci]['windows'][wi]['raw_features'] = None
                x = torch.stack(tensors, dim=0).to(device, non_blocking=True)
                with torch.no_grad():
                    _, output = model(x)
                output = output.detach().cpu()
                for j, key in enumerate(batch_jobs):
                    scores[key] = output[j]
                del raws, tensors, x, output

    for ci, result in enumerate(contig_results):
        contig_name = result['contig_name']
        contig_len = result['contig_len']
        bp_pred = []
        point_pred = []
        pred = []
        margin_pred = []
        chimeric_pred = []
        window_records = []

        if result.get('low_coverage'):
            # Low coverage is insufficient evidence, not a misassembly call.
            # Preserve the contig and mark it as abstained in compact outputs.
            pred.append(0)
            contig_test[contig_name] = {
                'bp_pred': bp_pred,
                'point_pred': point_pred,
                'max_prediction': max(pred, default=0),
                'max_margin': 0.0,
                'top3_margin_mean': 0.0,
                'chimeric_pred': chimeric_pred,
                'contig_len': contig_len,
                'window_prediction': pred,
                'window_records': window_records,
                'n_scored_windows': 0,
                'low_coverage': True,
            }
            continue

        windows = result['windows']
        for wi, w in enumerate(windows):
            point = w['point']
            key = (ci, wi)
            if w['skip_score'] or key not in scores:
                point_pred.append((point, 0))
                pred.append(0)
                window_records.append([
                    int(point), 0.0, 0.0, 0.0,
                    int(w.get('max_bp', point)), False, True, False
                ])
                continue
            out = scores[key]
            max_value, max_index = torch.max(out, dim=0)
            logit0 = float(out[0].item())
            logit1 = float(out[1].item())
            margin = float(logit1 - logit0)
            p0 = float(1.0 / (1.0 + np.exp(-logit0)))
            p1 = float(1.0 / (1.0 + np.exp(-logit1)))
            predicted_positive = max_index.item() == 1
            window_ts = w['window_ts']
            has_chimeric_support = bool(
                window_ts is not None and (not np.all(window_ts == 0))
            )
            # Chimeric probability = softmax(logits)[1] = sigmoid(logit1 - logit0).
            p_chimeric = float(1.0 / (1.0 + np.exp(-margin)))
            point_pred.append((point, p_chimeric))
            pred.append(p_chimeric)
            margin_pred.append(margin)
            window_records.append([
                int(point), p0, p1, margin, int(w['max_bp']),
                has_chimeric_support, False, predicted_positive
            ])
            if predicted_positive and p_chimeric >= score_cut:
                bp_pred.append((w['max_bp'], p_chimeric))
                if has_chimeric_support:
                    chimeric_pred.append((w['max_bp'], p_chimeric))

        top_margins = sorted(margin_pred, reverse=True)[:3]
        contig_test[contig_name] = {
            'bp_pred': bp_pred,
            'point_pred': point_pred,
            'max_prediction': max(pred, default=0),
            'max_margin': max(margin_pred, default=0),
            'top3_margin_mean': (
                float(np.mean(top_margins)) if top_margins else 0.0
            ),
            'chimeric_pred': chimeric_pred,
            'contig_len': contig_len,
            'window_prediction': pred,
            'window_records': window_records,
            'n_scored_windows': len(margin_pred),
            'low_coverage': False,
        }

    return contig_test

def _extract_contig_chunk(extract_fn, names, contig_threads):
    if contig_threads <= 1 or len(names) <= 1:
        return [extract_fn(n) for n in names]
    workers = min(contig_threads, len(names))
    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(processes=workers) as pool:
        return list(pool.imap(extract_fn, names, chunksize=1))


def iter_contig_chunks(total_contig_name, contig_chunk_size, max_windows_per_chunk,
                       extract_fn, contig_threads):
    """Yield contig result chunks with bounded live feature memory."""
    i = 0
    n = len(total_contig_name)
    pbar = tqdm(total=n, desc='Streaming extract+infer ...')
    while i < n:
        chunk_names = []
        while i < n and len(chunk_names) < contig_chunk_size:
            chunk_names.append(total_contig_name[i])
            i += 1

        results = _extract_contig_chunk(extract_fn, chunk_names, contig_threads)

        sub = []
        sub_windows = 0
        for result in results:
            w = _count_infer_windows(result)
            if sub and (sub_windows + w > max_windows_per_chunk):
                yield sub
                sub = []
                sub_windows = 0
            if (not sub) and w > max_windows_per_chunk:
                yield [result]
                pbar.update(1)
                continue
            sub.append(result)
            sub_windows += w
            pbar.update(1)
        if sub:
            yield sub
    pbar.close()


def _compact_checkpoint_record(contig_name, rec):
    return {
        'contig_name': contig_name,
        'contig_len': int(rec['contig_len']),
        'bp_pred': [[int(a), float(b)] for a, b in rec.get('bp_pred', [])],
        'chimeric_pred': [[int(a), float(b)] for a, b in rec.get('chimeric_pred', [])],
        'max_prediction': float(rec.get('max_prediction', 0)),
        'max_margin': float(rec.get('max_margin', 0)),
        'top3_margin_mean': float(rec.get('top3_margin_mean', 0)),
        'n_scored_windows': int(rec.get('n_scored_windows', 0)),
        'low_coverage': bool(rec.get('low_coverage', False)),
    }


def load_prediction_checkpoint(checkpoint_path):
    contig_test = {}
    if not os.path.isfile(checkpoint_path):
        return contig_test
    n_ok, n_bad = 0, 0
    with open(checkpoint_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                name = obj['contig_name']
                contig_test[name] = {
                    'bp_pred': [(int(a), float(b)) for a, b in obj.get('bp_pred', [])],
                    'chimeric_pred': [(int(a), float(b)) for a, b in obj.get('chimeric_pred', [])],
                    'max_prediction': float(obj.get('max_prediction', 0)),
                    'max_margin': float(obj.get('max_margin', 0)),
                    'top3_margin_mean': float(obj.get('top3_margin_mean', 0)),
                    'n_scored_windows': int(obj.get('n_scored_windows', 0)),
                    'low_coverage': bool(obj.get('low_coverage', False)),
                    'contig_len': int(obj['contig_len']),
                    'point_pred': [],
                    'window_prediction': [],
                }
                n_ok += 1
            except Exception:
                n_bad += 1
    msg = f"Loaded checkpoint: {n_ok} contigs from {checkpoint_path}"
    if n_bad:
        msg += f" (skipped {n_bad} bad lines)"
    print(msg)
    return contig_test


def append_prediction_checkpoint(checkpoint_path, contig_preds):
    parent = os.path.dirname(checkpoint_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(checkpoint_path, 'a') as f:
        for contig_name, rec in contig_preds.items():
            f.write(json.dumps(_compact_checkpoint_record(contig_name, rec), separators=(',', ':')) + '\n')
        f.flush()
        os.fsync(f.fileno())


def append_window_predictions(window_path, contig_preds):
    """Persist all lightweight window scores for post-hoc aggregation/calibration.

    Record columns are:
    [window_center, class0_score, class1_score, logit_margin,
     candidate_breakpoint, has_chimeric_support, skipped,
     predicted_positive_class]
    """
    parent = os.path.dirname(window_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(window_path, 'a') as f:
        for contig_name, rec in contig_preds.items():
            obj = {
                'contig_name': contig_name,
                'contig_len': int(rec['contig_len']),
                'n_scored_windows': int(rec.get('n_scored_windows', 0)),
                'max_margin': float(rec.get('max_margin', 0)),
                'top3_margin_mean': float(rec.get('top3_margin_mean', 0)),
                'low_coverage': bool(rec.get('low_coverage', False)),
                'columns': [
                    'point', 'p0', 'p1', 'margin', 'max_bp',
                    'chimeric_support', 'skipped', 'predicted_positive'
                ],
                'windows': rec.get('window_records', []),
            }
            f.write(json.dumps(obj, separators=(',', ':')) + '\n')
        f.flush()
        os.fsync(f.fileno())


def write_prediction_tsv(tsv_path, total_contig_name, contig_test):
    parent = os.path.dirname(tsv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = tsv_path + '.tmp'
    with open(tmp_path, 'w') as f:
        f.write('Assembly\tBreakPoint\tPrediction\tLength\tChimeric_BreakPoint\tChimeric_Prediction\n')
        for contig_name in total_contig_name:
            if contig_name not in contig_test:
                continue
            rec = contig_test[contig_name]
            if len(rec['bp_pred']) > 0:
                sorted_data = sorted(rec['bp_pred'], key=lambda x: x[1])
                bp = sorted_data[-1][0]
                bp_prediction = sorted_data[-1][1]
                if len(rec['chimeric_pred']) > 0:
                    sorted_chi = sorted(rec['chimeric_pred'], key=lambda x: x[1])
                    chi_point = sorted_chi[-1][0]
                    chi_prediction = sorted_chi[-1][1]
                    f.write(f"{contig_name}\t{bp}\t{bp_prediction}\t{rec['contig_len']}\t{chi_point}\t{chi_prediction}\n")
                else:
                    f.write(f"{contig_name}\t{bp}\t{bp_prediction}\t{rec['contig_len']}\t-1\t0\n")
            else:
                # float() keeps the TSV byte-identical between a fresh run
                # (abstained contig carries an int 0) and a resumed run (the
                # checkpoint round-trip yields 0.0). Idempotence matters here
                # because long jobs are routinely resumed after a timeout.
                f.write(f"{contig_name}\t-1\t{float(rec['max_prediction'])}\t{rec['contig_len']}\t-1\t0\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, tsv_path)


def predict(args):
    """Run inference only: extract window features and write per-contig
    prediction TSV (plus checkpoints). No contig correction is performed here.
    """
    contig_threads = args.contig_threads
    device = args.gpu_device
    fine_tune = args.fine_tune
    folder_path = args.folder_path
    model_weight_path = args.model_weight_path
    file_name = args.file_name
    contig_file_name = args.contig_file_name
    bam_file_name = args.bam_file_name
    score_cut = float(args.threshold)
    config['window_threads'] = args.window_threads
    config['max_contig_length'] = args.max_contig_length
    batch_size = int(getattr(args, 'infer_batch_size', 32))
    contig_chunk_size = int(getattr(args, 'contig_chunk_size', 8))
    max_windows_per_chunk = int(getattr(args, 'max_windows_per_chunk', 1024))
    resume = bool(getattr(args, 'resume', True))
    output_dir = getattr(args, 'output_dir', '') or ''
    checkpoint_file = getattr(args, 'checkpoint_file', 'checkpoint_predictions.jsonl')

    contig_path = os.path.join(folder_path, contig_file_name)
    align_path = os.path.join(folder_path, bam_file_name)
    ensure_indexes(contig_path, align_path)

    output_folder_path = output_dir if output_dir else f'{folder_path}/deepmm_output'
    os.makedirs(output_folder_path, exist_ok=True)
    checkpoint_path = os.path.join(output_folder_path, checkpoint_file)
    window_path = os.path.join(output_folder_path, 'window_predictions.jsonl')
    partial_tsv_path = os.path.join(output_folder_path, 'assembly_prediction.partial.tsv')
    final_tsv_path = os.path.join(output_folder_path, file_name)

    total_contig_name = []
    with pysam.FastaFile(contig_path) as contig_file:
        for contig_name in contig_file.references:
            contig_seq = contig_file.fetch(contig_name)
            if config['min_contig_length'] <= len(contig_seq) < config['max_contig_length']:
                total_contig_name.append(contig_name)

    print(f"Contigs to process: {len(total_contig_name)} (threads={contig_threads})")
    print(
        f"Streaming mode: contig_chunk_size={contig_chunk_size}, "
        f"max_windows_per_chunk={max_windows_per_chunk}, infer_batch_size={batch_size}"
    )
    print(f"Output/persist dir: {output_folder_path}")
    print(f"Checkpoint: {checkpoint_path} (resume={resume})")

    contig_test = {}
    if resume:
        contig_test = load_prediction_checkpoint(checkpoint_path)
    else:
        for p in (checkpoint_path, window_path, partial_tsv_path):
            if os.path.isfile(p):
                os.remove(p)

    done = set(contig_test.keys())
    pending_contig_name = [n for n in total_contig_name if n not in done]
    print(f"Resume status: done={len(done)}, pending={len(pending_contig_name)}")

    if pending_contig_name:
        print("Loading model for streamed GPU inference ...")
        weight_state = torch.load(model_weight_path, map_location=device, weights_only=True)
        arch = infer_arch_from_state_dict(weight_state)
        print(f"Detected backbone: {arch} ({model_weight_path})")
        model = LinearEvaluation(ResNetSimCLR(base_model=arch, out_dim=128))
        if fine_tune:
            print("Using Fine-tune model")
            model = FinetuneModel(model)
        else:
            print("Using Default model")

        model.load_state_dict(weight_state)
        model = model.to(device)
        model.eval()

        extract_fn = partial(extract_windows_for_contig, contig_path=contig_path, align_path=align_path)
        n_chunks = 0
        n_windows = 0

        for chunk_results in iter_contig_chunks(
            pending_contig_name,
            contig_chunk_size,
            max_windows_per_chunk,
            extract_fn,
            contig_threads,
        ):
            n_chunks += 1
            n_windows += sum(_count_infer_windows(r) for r in chunk_results)
            chunk_pred = run_batched_inference_chunk(
                model, device, chunk_results, score_cut, batch_size
            )
            append_prediction_checkpoint(checkpoint_path, chunk_pred)
            append_window_predictions(window_path, chunk_pred)
            for rec in chunk_pred.values():
                rec.pop('window_records', None)
                rec['point_pred'] = []
                rec['window_prediction'] = []
            contig_test.update(chunk_pred)
            write_prediction_tsv(partial_tsv_path, total_contig_name, contig_test)
            print(
                f"Checkpointed chunk {n_chunks}: +{len(chunk_pred)} contigs, "
                f"total_done={len(contig_test)}/{len(total_contig_name)}"
            )
            del chunk_results, chunk_pred
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"Finished streaming: chunks={n_chunks}, scored_windows~{n_windows}")
    else:
        print("All contigs already in checkpoint; skipping extract/infer.")

    missing = [n for n in total_contig_name if n not in contig_test]
    if missing:
        raise RuntimeError(
            f"Still missing {len(missing)} contig predictions after streaming "
            f"(e.g. {missing[:3]}). Refusing to write final outputs."
        )

    write_prediction_tsv(final_tsv_path, total_contig_name, contig_test)
    write_prediction_tsv(partial_tsv_path, total_contig_name, contig_test)

    if getattr(args, 'escore', True):  # E-value calibration is the final output (always applied)
        escore_name = getattr(args, 'escore_file_name', 'assembly_prediction.escore.tsv')
        escore_path = os.path.join(output_folder_path, escore_name)
        try:
            _, summary = run_escore(
                output_folder_path,
                out_tsv=escore_path,
                file_name=file_name,
                mode=getattr(args, 'escore_mode', 'calibrated'),
                r=getattr(args, 'escore_r', 100),
                floor=getattr(args, 'escore_floor', None),
            )
        except Exception as exc:
            # The raw TSV is already on disk and valid; do not lose a long run
            # over a failure in the optional E-value step.
            print(f"WARNING: E-value step skipped ({type(exc).__name__}: {exc})")
        else:
            print(f"E-value TSV: {escore_path}")
            print(f"  {describe_escore(summary)}")
            print(f"  `correct` picks this file up automatically; "
                  f"threshold {E_P05} corresponds to p < 0.05")

    print(f"Prediction complete. Outputs in {output_folder_path}")
