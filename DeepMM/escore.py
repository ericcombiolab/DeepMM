#!/usr/bin/env python
"""Label-free E-value calibration for DeepMM contig-level predictions.

CONTEXT / WHY THIS EXISTS
-------------------------
`DeepMM predict` scores every sliding window of a contig independently and then
reduces the window scores to ONE contig score.  The reduction it uses natively is
the MAXIMUM window margin (``max_margin``), exposed in ``assembly_prediction.tsv``
as the ``Prediction`` column.

A maximum is a poor contig-level statistic here, for two reasons.

1.  It grows with the number of windows, so long contigs are systematically
    pushed towards "misassembled" even when no single window is convincing.  On
    the CAMI evaluation the raw contig score correlates POSITIVELY with contig
    length among labeled negatives: Spearman +0.395 (metaSPAdes) / +0.144
    (megahit).

2.  The margin distribution is strongly length-dependent, so no single pooled
    threshold can serve all contigs.  The rate of windows with margin >= 4 is
    0.0038 for 5-20 kb contigs but 0.00012 for >= 150 kb contigs -- a 32x gap
    (10 megahit samples, 1.16M scored windows).

Together these make the contig score a function of contig length, which is
exactly what an E-value against an empirical null is supposed to remove.

WHAT THIS MODULE DOES
---------------------
It converts the contig's max margin into an E-value = ``-log10(p)`` against an
EMPIRICAL NULL built from other windows of the SAME sample.  Nothing else
changes: the model, the features and the window scores are untouched.  Only the
single number that ``correct`` compares against ``--threshold`` is replaced --
plus the ``BreakPoint`` column, which is blanked for contigs the E-value would
not send to ``correct`` (see BREAKPOINT below).

Two nulls are provided.

``pooled``
    Null = the margins of EVERY scored window of the sample (all contigs pooled).

        q(t) = fraction of pooled windows with margin >= t
        p    = 1 - (1 - q)**n          # n = scored windows of this contig
        E    = -log10(p)

    ``q`` is floored at ``1/N`` by default, so a contig whose max exceeds every
    pooled window cannot reach p = 0 (and hence an infinite E).

    This null is genuinely label-free and cheap, but its deep tail is dominated
    by SHORT contigs, which are the noisiest: exceedance at margin >= 4 is 32x
    higher for 5-20 kb contigs than for >= 150 kb contigs.  Asking a long contig
    to beat that threshold is asking it to beat a distribution it essentially
    never produces.  The outcome is not a length bias but an over-correction:
    Spearman(E, length) on labeled negatives is -0.530 (metaSPAdes) / -0.580
    (megahit), i.e. long contigs now score too LOW.

``calibrated``   (default)
    Null = a LEAVE-ONE-OUT, CONTIG-LEVEL max null restricted to comparable
    contigs of the same sample.  For a target contig i with max margin ``t_i``
    and ``n_i`` scored windows:

        R_i    = { other contigs c of the SAME sample with n_c >= r }
        F_c    = ECDF of contig c's kept window margins
        q_c(t) = 1 - F_c(t)
        p_i    = mean_{c in R_i} [ 1 - (1 - q_c(t_i)) ** n_i ]
        E      = -log10(p_i)

    ``r`` is a minimum-reference-window-count match (default 100, i.e. reference
    contigs of roughly >= 41 kb at the shipped window stride).  A contig is now
    compared against the *typical best-window* behaviour of other contigs with
    similar evidence, instead of against a pooled marginal.

    The self-contribution is removed (leave one out): for a target that is
    itself a reference, ``t_k`` is the maximum of its own windows so
    ``F_k(t_k) = 1``, ``q_self = 1/n_k`` exactly, and that term is subtracted
    from both numerator and denominator.  No floor is applied by default --
    flooring ``q_c`` at ``1/n_c`` destroys the calibration: at r=100 on six
    megahit samples it moves the median negative p from 0.516 to 0.629, the KS
    distance to Uniform from 0.088 to 0.174, and drags the length correlation
    back out to -0.580.  It remains reachable via ``floor=True`` for ablation
    only.

    If a sample has fewer than 2 reference contigs the calibrated null is
    undefined and this module falls back to ``pooled``.

WHY r IS NOT A TUNED PARAMETER
------------------------------
Calibration improves monotonically in ``r``, so there is no interior optimum to
fit.  On six small/medium megahit samples (r=10/25/50/100):

    median p, labeled negatives   0.631 / 0.585 / 0.552 / 0.516
    KS(p, Uniform), negatives     0.215 / 0.151 / 0.109 / 0.088
    Spearman(E, length), negatives +0.070 / -0.080 / -0.166 / -0.233

Only the length correlation is non-monotone (it crosses zero near r=25 and
slightly overshoots), so ``r`` trades a little residual length dependence against
better p-value calibration -- it is not fitted to a metric peak.  ``r = 100`` is
the largest value in the swept grid; it still leaves these samples with hundreds
of reference contigs (median 316), and a sample with fewer than 2 falls back to
``pooled`` automatically.

LABEL-FREE-NESS
---------------
No ground-truth file is read anywhere.  The null is built exclusively from the
model's own window margins, within a single sample.  The one tunable, ``r``, is a
window-count match, and the E threshold commonly used (``E_P05`` = 1.30103,
equivalent to p < 0.05) is a conventional significance level, not a fitted
operating point.

MEASURED EFFECT  (CAMI evaluation, labeled contigs >= 5 kb, as cached in
evaluation/CAMI/analysis/bpfilter_contig_scores_<assembler>.tsv.gz)

    rho_neg/rho_pos  Spearman(score, contig length) over labeled negatives and
                     positives; 0 is ideal.
    med_p_neg        median p (= 10**-E) over labeled negatives; ~0.5 is ideal.
    KS_neg           KS distance from those p to Uniform(0,1); 0 is ideal.

                   metaSPAdes   14,771 contigs   12,555 neg / 2,216 pos
    variant        rho_neg  rho_pos  med_p_neg  KS_neg   AUROC    AUPRC
    raw max         +0.395   -0.008      0.126   0.686  0.7720   0.4946
    pooled E        -0.530   -0.511      0.888   0.407  0.8263   0.5756
    calibrated E    -0.088   -0.367      0.531   0.047  0.8488   0.5683

                   megahit      28,718 contigs   11,978 neg / 16,740 pos
    raw max         +0.144   +0.007      0.131   0.594  0.7011   0.7750
    pooled E        -0.580   -0.496      0.870   0.374  0.7692   0.8160
    calibrated E    -0.278   -0.376      0.489   0.047  0.8026   0.8268

Reading: the raw max is length-inflated (+rho); the pooled null overshoots the
correction (-rho, and its p-values are far from Uniform, KS 0.37-0.41); the
calibrated null removes most of the length signal and is by far the best
calibrated (KS 0.047 on BOTH assemblers).  AUROC improves monotonically from
raw -> pooled -> calibrated on both assemblers.  AUPRC improves too, except on
metaSPAdes where pooled is marginally ahead (0.5756 vs 0.5683): AUPRC rewards
the extra positives pooled flags, so the two variants trade a little precision
for a better-calibrated ranking.  AUROC is the cleaner headline here because
AUPRC has no fixed baseline across samples of differing prevalence.

COST  (largest CAMI sample, high/sample_1: 81,364 contigs, 2.95M scored windows,
305 MB of window_predictions.jsonl)
    load_window_stats   6.6 s
    calibrated r=100   44.4 s   (5,671 reference contigs)
    full compute()     52.2 s
So the step costs well under a minute at the end of a multi-hour predict job.

ABSTENTION
----------
``predict`` marks a contig ``low_coverage`` (fewer than 10 reads) and reports a
raw score of 0: it has no evidence, so it declines to call anything.  A contig
with zero scored windows must therefore stay at ``abstain_score`` (default 0.0),
NOT be mapped to the top of the E scale.  The naive formula would send it to
E = 300 (because ``1 - (1-q)**0 = 0``, so ``p = 0``); this module guards against
that explicitly.  The earlier standalone ``escore_post2.py`` did not, which
inflated 19 (metaSPAdes) / 94 (megahit) abstained contigs to E = 300.

BREAKPOINT
----------
``BreakPoint`` is the coordinate ``correct`` would split at, so it is only
meaningful for a contig that is actually going to be split.  This module writes
one ONLY when ``E >= bp_min_e`` (default ``E_P05`` = 1.30103); every other row
keeps ``-1``.  That mirrors the raw TSV, where ``predict`` leaves the column at
``-1`` whenever a window misses its ``score_cut`` -- previously the E-value path
overwrote it unconditionally, advertising splits that would never happen.

The gate uses ``>=`` while ``correct`` splits on ``>``, i.e. it is deliberately
one notch looser, so a contig ``correct`` wants to split can never be dropped
for lack of a breakpoint.  The flip side: keep ``correct --threshold`` at or
above ``bp_min_e``, otherwise contigs falling in between will silently stay
unsplit.
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd

MODES = ("calibrated", "pooled")
DEFAULT_MODE = "calibrated"
DEFAULT_R = 100
DEFAULT_ABSTAIN_SCORE = 0.0

# E-value equivalent of p < 0.05 / p < 0.01, i.e. -log10(p).
E_P05 = round(-math.log10(0.05), 5)      # 1.30103
E_P01 = round(-math.log10(0.01), 5)      # 2.0
E_FLOOR_P = 1e-300                        # caps E at 300


def default_floor(mode):
    """Floored ``q`` for the pooled null (legacy behaviour); no floor for the
    calibrated null (flooring breaks its calibration)."""
    return mode == "pooled"


def load_window_stats(window_path):
    """Parse ``window_predictions.jsonl`` written by ``predict``.

    Returns ``{contig_name: {'n', 't', 'bp', 'margins'}}`` where

        n       scored windows (``n_scored_windows``)
        t       contig max margin (``max_margin``)
        bp      candidate breakpoint of the argmax-margin window (-1 if none)
        margins sorted float array of the kept (non-skipped) window margins
    """
    stats = {}
    if not os.path.isfile(window_path):
        return stats
    with open(window_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            name = o["contig_name"]
            n = int(o.get("n_scored_windows", 0))
            margins = []
            best = -np.inf
            bp = -1
            for w in o.get("windows", []):
                if len(w) < 8:
                    continue
                if w[6]:                        # skipped window
                    continue
                m = float(w[3])
                margins.append(m)
                if m > best:
                    best = m
                    bp = int(w[4])
            margins = np.sort(np.asarray(margins, dtype=float))
            # ``max_margin`` is the authoritative contig score; fall back to the
            # maximum kept window margin if an older record lacks it.
            t = float(o.get("max_margin", best if margins.size else 0.0))
            stats[name] = {"n": n, "t": t, "bp": bp, "margins": margins}
    return stats


def _pooled(names, stats, n_arr, t_arr, floor, abstain_score):
    pool = np.concatenate([stats[c]["margins"] for c in names]) if names else np.empty(0)
    if pool.size == 0:
        return np.full(len(names), abstain_score, dtype=float)
    pool = np.sort(pool)
    N = pool.size
    F = np.searchsorted(pool, t_arr, side="left") / N
    q = 1.0 - F
    if floor:
        q = np.maximum(q, 1.0 / N)
    p = 1.0 - (1.0 - q) ** n_arr
    e = -np.log10(np.maximum(p, E_FLOOR_P))
    e[n_arr <= 0] = abstain_score
    return e


def _calibrated(names, stats, n_arr, t_arr, r, floor, abstain_score):
    ref = np.nonzero(n_arr >= r)[0]
    n_ref = ref.size
    if n_ref < 2:
        return None, 0, True                       # caller falls back to pooled

    acc = np.zeros(len(names), dtype=np.float64)
    for j in ref:
        M = stats[names[int(j)]]["margins"]
        nc = n_arr[int(j)]
        if M.size == 0:
            continue
        q = 1.0 - np.searchsorted(M, t_arr, side="left") / nc
        if floor:
            q = np.maximum(q, 1.0 / nc)
        acc += 1.0 - (1.0 - q) ** n_arr

    # Leave-one-out: drop each reference contig's contribution to itself.
    # Its own max makes q_self = 1/n_k exactly.
    nc_r = n_arr[ref]
    acc[ref] -= 1.0 - (1.0 - 1.0 / nc_r) ** n_arr[ref]

    denom = np.full(len(names), float(n_ref))
    denom[ref] -= 1.0
    p = acc / np.maximum(denom, 1.0)
    e = -np.log10(np.maximum(p, E_FLOOR_P))
    e[n_arr <= 0] = abstain_score
    return e, n_ref, False


def compute(out_dir, file_name="assembly_prediction.tsv", mode=DEFAULT_MODE,
            r=DEFAULT_R, floor=None, abstain_score=DEFAULT_ABSTAIN_SCORE,
            bp_min_e=E_P05):
    """Recompute the ``Prediction`` column of a prediction TSV as an E-value.

    Returns ``(df, summary)``.  ``df`` is the input TSV with ``Prediction`` (and
    ``BreakPoint``, when present) replaced.  Contigs that have no entry in
    ``window_predictions.jsonl`` keep the abstain score.

    ``BreakPoint`` is only written for contigs with ``E >= bp_min_e``; every
    other row gets ``-1``, so the column never advertises a split that
    ``correct`` would not perform.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if floor is None:
        floor = default_floor(mode)

    window_path = os.path.join(out_dir, "window_predictions.jsonl")
    tsv_path = os.path.join(out_dir, file_name)
    if not os.path.isfile(window_path):
        raise SystemExit(f"missing {window_path}")
    if not os.path.isfile(tsv_path):
        raise SystemExit(f"missing {tsv_path}")

    stats = load_window_stats(window_path)
    df = pd.read_csv(tsv_path, sep="\t")
    names = [str(c) for c in df["Assembly"]]

    have = np.asarray([i for i, c in enumerate(names) if c in stats], dtype=int)
    if have.size == 0:
        raise SystemExit(f"no contig in {tsv_path} has windows in {window_path}")

    sub = [names[i] for i in have]
    n_arr = np.asarray([stats[c]["n"] for c in sub], dtype=float)
    t_arr = np.asarray([stats[c]["t"] for c in sub], dtype=float)

    n_ref = 0
    fallback = False
    esub = None
    if mode == "calibrated":
        esub, n_ref, fallback = _calibrated(sub, stats, n_arr, t_arr, r, floor, abstain_score)
    if esub is None:
        if mode == "calibrated":
            fallback = True
        esub = _pooled(sub, stats, n_arr, t_arr, floor, abstain_score)

    e_full = np.full(len(names), float(abstain_score), dtype=float)
    bp_full = np.full(len(names), -1, dtype=int)
    for pos, i in enumerate(have):
        e_full[i] = esub[pos]
        # Only a contig that clears the decision cut gets a coordinate.  Note
        # ``>=`` here vs ``>`` in ``correct``: one notch looser on purpose, so
        # this gate can never hide a contig ``correct`` wants to split.
        if e_full[i] >= bp_min_e:
            bp_full[i] = stats[names[i]]["bp"]

    df["Prediction"] = e_full
    if "BreakPoint" in df.columns:
        df["BreakPoint"] = bp_full

    summary = {
        "mode": mode,
        "r": r,
        "floor": bool(floor),
        "rows_in_tsv": len(df),
        "contigs_scored": int(have.size),
        "contigs_abstained": int(len(names) - have.size),
        "zero_window_contigs": int((n_arr <= 0).sum()),
        "reference_contigs": int(n_ref),
        "pooled_fallback": bool(fallback),
        "n_at_or_above_p05": int((e_full >= E_P05).sum()),
        "bp_min_e": float(bp_min_e),
        "breakpoints_reported": int((bp_full >= 0).sum()),
    }
    return df, summary


def run(out_dir, out_tsv=None, file_name="assembly_prediction.tsv",
        mode=DEFAULT_MODE, r=DEFAULT_R, floor=None,
        abstain_score=DEFAULT_ABSTAIN_SCORE, bp_min_e=E_P05):
    """Compute and write the E-value TSV.  Returns ``(path, summary)``."""
    df, summary = compute(out_dir, file_name=file_name, mode=mode, r=r,
                          floor=floor, abstain_score=abstain_score,
                          bp_min_e=bp_min_e)
    if out_tsv is None:
        out_tsv = os.path.join(out_dir, "assembly_prediction.escore.tsv")
    parent = os.path.dirname(os.path.abspath(out_tsv))
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_csv(out_tsv, sep="\t", index=False)
    return out_tsv, summary


def describe(summary):
    """One-line human-readable summary of a ``compute``/``run`` result."""
    bits = [
        f"mode={summary['mode']}",
        f"r={summary['r']}",
        f"floor={int(summary['floor'])}",
        f"contigs={summary['contigs_scored']}",
        f"zero_window={summary['zero_window_contigs']}",
        f"E>={E_P05}: {summary['n_at_or_above_p05']}",
        f"bp>={summary['bp_min_e']:g}: {summary['breakpoints_reported']}",
    ]
    if summary["pooled_fallback"]:
        bits.append("POOLED_FALLBACK")
    return " ".join(bits)


def _main():
    ap = argparse.ArgumentParser(
        description="Rewrite a DeepMM prediction TSV with label-free E-values")
    ap.add_argument("--input_dir", required=True,
                    help="folder holding window_predictions.jsonl + the prediction TSV")
    ap.add_argument("--out_tsv", default="",
                    help="output TSV path [default: <input_dir>/assembly_prediction.escore.tsv]")
    ap.add_argument("--file_name", default="assembly_prediction.tsv",
                    help="input prediction TSV name inside --input_dir")
    ap.add_argument("--mode", default=DEFAULT_MODE, choices=MODES)
    ap.add_argument("--r", type=int, default=DEFAULT_R,
                    help="minimum reference-contig window count for the calibrated null")
    ap.add_argument("--floor", dest="floor", action="store_true", default=None,
                    help="floor q (default: on for pooled, off for calibrated)")
    ap.add_argument("--no_floor", dest="floor", action="store_false",
                    help="never floor q")
    ap.add_argument("--abstain_score", type=float, default=DEFAULT_ABSTAIN_SCORE,
                    help="score for contigs with no scored windows [default: 0]")
    ap.add_argument("--bp_min_e", type=float, default=E_P05,
                    help="minimum E-value for writing a BreakPoint; below it the column "
                         "stays -1 [default: 1.30103].  Keep it <= the "
                         "`correct --threshold` you intend to use")
    a = ap.parse_args()
    path, summary = run(a.input_dir, out_tsv=(a.out_tsv or None), file_name=a.file_name,
                        mode=a.mode, r=a.r, floor=a.floor, abstain_score=a.abstain_score,
                        bp_min_e=a.bp_min_e)
    print(f"wrote {path}")
    print(describe(summary))


if __name__ == "__main__":
    _main()
