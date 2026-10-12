import os
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

from .tools import config
from .escore import E_P05

RAW_TSV = 'assembly_prediction.tsv'
ESCORE_TSV = 'assembly_prediction.escore.tsv'
ESCORE_DIR = 'escore'          # legacy escore_post.py layout


def escore_candidates(output_folder_path):
    """Every location an E-value TSV is known to live, most specific first.

    Both layouts are in use, and the second one cannot be found by name alone
    because it reuses the raw TSV's file name:

        <out>/assembly_prediction.escore.tsv    written by `predict` / `escore`
        <out>/escore/assembly_prediction.tsv    written by the older
                                                escore_post.py + correct flow
        <out>/../escore/assembly_prediction.tsv same flow, when <out> is the
                                                deepmm_output folder
    """
    out = os.path.abspath(output_folder_path)
    return [
        os.path.join(out, ESCORE_TSV),
        os.path.normpath(os.path.join(out, ESCORE_DIR, RAW_TSV)),
        os.path.normpath(os.path.join(out, os.pardir, ESCORE_DIR, RAW_TSV)),
    ]


def resolve_prediction_tsv(output_folder_path, file_name='', source='auto'):
    """Pick which prediction TSV ``correct`` acts on.

    ``file_name`` wins when given explicitly.  Otherwise ``source`` decides:

        raw     -> assembly_prediction.tsv   (raw score, a sigmoid in (0, 1))
        escore  -> the E-value TSV, in whichever layout it was written
        auto    -> the E-value TSV when one exists, else the raw TSV
    """
    if file_name:
        return os.path.join(output_folder_path, file_name)
    raw = os.path.join(output_folder_path, RAW_TSV)
    if source == 'raw':
        return raw
    candidates = escore_candidates(output_folder_path)
    found = next((c for c in candidates if os.path.isfile(c)), None)
    if found:
        return found
    # Nothing on disk: return the canonical name for --input escore so the
    # caller's error message points at the file we would have written.
    return candidates[0] if source == 'escore' else raw


def score_scale(tsv_path):
    """Decide whether a prediction TSV holds E-values or raw scores.

    The two live on different scales, so the sensible default cutoff depends on
    which one this is.  Contents win over naming, because the legacy layout
    reuses the raw TSV's file name: an E-value is ``-log10(p)`` and is unbounded
    above, whereas a raw score is a sigmoid and can never exceed 1, so a maximum
    above 1 settles it outright.  Otherwise fall back to the path hints.
    """
    try:
        col = pd.read_csv(tsv_path, sep='\t', usecols=['Prediction'])['Prediction']
        col = pd.to_numeric(col, errors='coerce').dropna()
        if len(col) and float(col.max()) > 1.0:
            return 'escore'
    except Exception:
        pass
    if os.path.basename(tsv_path).endswith('.escore.tsv'):
        return 'escore'
    if os.path.basename(os.path.dirname(os.path.abspath(tsv_path))) == ESCORE_DIR:
        return 'escore'
    return 'raw'


def default_threshold(tsv_path):
    """E-values and raw scores live on different scales, so the sensible default
    cutoff depends on which score the TSV holds."""
    return E_P05 if score_scale(tsv_path) == 'escore' else 0.9


def correct(args):
    """Split misassembled contigs using an existing assembly_prediction.tsv.

    This is a pure post-processing step: it reads the prediction TSV produced
    by ``predict`` (or a prior ``correct`` run) and does NOT run any inference.

    Correction uses the generic ``Prediction`` column (all misassembly types),
    not the ``Chimeric_Prediction`` column (which was an interspecies-specific
    heuristic gated on ``window_ts != 0``). This is the new intended logic.
    """
    folder_path = args.folder_path
    contig_file_name = args.contig_file_name
    bam_file_name = getattr(args, 'bam_file_name', 'sort.bam')  # unused; kept for CLI compat
    file_name = getattr(args, 'file_name', '') or ''
    output_dir = getattr(args, 'output_dir', '') or ''
    source = getattr(args, 'input', 'auto') or 'auto'

    contig_path = os.path.join(folder_path, contig_file_name)

    output_folder_path = output_dir if output_dir else f'{folder_path}/deepmm_output'
    os.makedirs(output_folder_path, exist_ok=True)
    final_tsv_path = resolve_prediction_tsv(output_folder_path, file_name, source)

    if not os.path.isfile(final_tsv_path):
        raise FileNotFoundError(
            f"Prediction TSV not found: {final_tsv_path}. "
            f"Run `predict` first (it writes both {RAW_TSV} and {ESCORE_TSV}), "
            f"or ensure --output_dir points at the folder holding them. "
            f"You can also run `escore` to build the E-value TSV from an existing run."
        )

    explicit_threshold = args.threshold is not None
    scale = score_scale(final_tsv_path)
    score_cut = float(args.threshold) if explicit_threshold else default_threshold(final_tsv_path)
    print(f"Using {final_tsv_path} | scale={scale} | threshold={score_cut}"
          f"{'' if explicit_threshold else ' (default for this score scale)'}")
    if scale == 'escore' and score_cut < E_P05:
        print(
            f"WARNING: threshold {score_cut} is below the E-value BreakPoint gate "
            f"({E_P05}). Contigs with {score_cut} < E < {E_P05} are selected here but "
            f"carry BreakPoint == -1, so they will be silently left unsplit. Keep the "
            f"threshold at or above {E_P05}, or regenerate the TSV with "
            f"`escore --bp_min_e {score_cut}`."
        )

    df = pd.read_csv(final_tsv_path, sep='\t')

    # New logic: use generic Prediction (all misassembly types).
    print('Correcting all misassemblies (Prediction column)')
    breakcontigs = df.loc[df['Prediction'] > score_cut, ]
    breakcontigs = breakcontigs.loc[breakcontigs['BreakPoint'] > config['min_split_length'], ]
    breakcontigs = breakcontigs.loc[(breakcontigs['Length'] - breakcontigs['BreakPoint']) > config['min_split_length'], ]
    breakcontigs = list(np.unique(breakcontigs['Assembly']))

    corrected_contig_file = os.path.join(output_folder_path, "corrected_contigs.fa")
    input_file = SeqIO.parse(contig_path, "fasta")

    with open(corrected_contig_file, "w") as corrected_file:
        for record in tqdm(input_file, desc='Correcting ... '):
            if record.id in breakcontigs:
                bp = int(df.loc[df['Assembly'] == record.id, 'BreakPoint'].values[0])
                corrected_file.write(">" + record.id + "_1\n")
                corrected_file.write(str(record.seq[:bp]) + "\n")
                corrected_file.write(">" + record.id + "_2\n")
                corrected_file.write(str(record.seq[bp:]) + "\n")
            else:
                corrected_file.write(">" + record.id + "\n")
                corrected_file.write(str(record.seq) + "\n")

    print(f"A total of {len(breakcontigs)} misassembled contigs are corrected")
    print(f"Final outputs in {output_folder_path}")
