import os
import sys
import argparse
import logging
import warnings

from .dataset_finetune import *
from .dataset_pretrain import *
from .finetune import *
from .prediction import *
from .correct import *
from .escore import (
    run as run_escore,
    describe as describe_escore,
    E_P05,
    MODES as ESCORE_MODES,
    DEFAULT_R as ESCORE_DEFAULT_R,
)


def get_opts(options):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Identify and correct metagenomic misassemblies with deep learning')

    subparsers = parser.add_subparsers(title='DeepMM subcommands',
                                       dest='cmd',
                                       metavar='')

    predict = subparsers.add_parser('predict', help='Identify misassemblies (inference only, writes prediction TSV)')
    predict.add_argument('--folder_path', default='', help='Path to data folder')
    predict.add_argument('--model_weight_path', default='./pretrain-model-weight/checkpoint.pt', help='Path to model weight')
    predict.add_argument('--fine_tune', action='store_true', help='Use Fine-tune model')
    predict.add_argument('--contig_threads', default=8, type=int, help='Maximum number of threads [default: 8]')
    predict.add_argument('--window_threads', default=8, type=int, help='Maximum number of threads [default: 8]')
    predict.add_argument('--gpu_device', default='cuda:0', help='GPU device id')
    predict.add_argument('--file_name', default='assembly_prediction.tsv', help='Result file name')
    predict.add_argument('--contig_file_name', default='final.contigs.fa', help='Result file name')
    predict.add_argument('--bam_file_name', default='sort.bam', help='Result file name')
    predict.add_argument('--threshold', default=0.9, type=float,
                         help='Score cutoff for window prediction')
    predict.add_argument('--infer_batch_size', default=32, type=int,
                         help='GPU batch size for window inference [default: 32]')
    predict.add_argument('--max_contig_length', default=float('inf'), type=float, help='Maximum contig length to process')
    predict.add_argument('--contig_chunk_size', default=8, type=int,
                         help='Contigs extracted together before GPU inference [default: 8]')
    predict.add_argument('--max_windows_per_chunk', default=1024, type=int,
                         help='Max scored windows kept in memory per chunk [default: 1024]')
    predict.add_argument('--output_dir', default='',
                         help='Persist output/checkpoint directory (prefer NFS path)')
    predict.add_argument('--checkpoint_file', default='checkpoint_predictions.jsonl',
                         help='JSONL checkpoint file name under output_dir')
    predict.add_argument('--resume', dest='resume', action='store_true', default=True,
                         help='Resume from checkpoint if present [default: True]')
    predict.add_argument('--no_resume', dest='resume', action='store_false',
                         help='Ignore existing checkpoint and start fresh')
    predict.add_argument('--escore_mode', default='calibrated', choices=list(ESCORE_MODES),
                         help='Empirical null used for the E-value')
    predict.add_argument('--escore_r', default=ESCORE_DEFAULT_R, type=int,
                         help='Minimum reference-contig window count (calibrated mode only)')
    predict.add_argument('--escore_floor', dest='escore_floor', action='store_true', default=None,
                         help='Floor q at 1/(pool size) [default: on for pooled, off for calibrated]')
    predict.add_argument('--escore_no_floor', dest='escore_floor', action='store_false',
                         help='Never floor q')
    predict.add_argument('--escore_file_name', default='assembly_prediction.escore.tsv',
                         help='E-value TSV file name written next to the raw TSV')

    correct = subparsers.add_parser('correct', help='Split misassembled contigs using existing prediction TSV')
    correct.add_argument('--folder_path', default='', help='Path to data folder (contains contig FASTA)')
    correct.add_argument('--contig_file_name', default='final.contigs.fa', help='Contig FASTA file name')
    correct.add_argument('--bam_file_name', default='sort.bam', help='BAM file name (unused; kept for CLI compat)')
    correct.add_argument('--file_name', default='',
                         help='Explicit prediction TSV name. When omitted, --input decides '
                              '(defaults to the E-value TSV if present)')
    correct.add_argument('--input', default='auto', choices=('auto', 'escore', 'raw'),
                         help='Which prediction TSV to act on when --file_name is omitted')
    correct.add_argument('--output_dir', default='',
                         help='Directory containing the prediction TSV and receiving corrected_contigs.fa')
    correct.add_argument('--threshold', default=None, type=float,
                         help='Score cutoff for contig splitting, applied to the Prediction column. '
                              f'Defaults to {E_P05} (p<0.05) for the E-value TSV and 0.9 for the raw '
                              'score TSV, which live on different scales. On an E-value TSV, keep it '
                              f'at or above {E_P05}: below that cutoff contigs can be selected while '
                              'their BreakPoint is still -1, and they are then dropped silently')

    escore = subparsers.add_parser(
        'escore',
        help='Recompute an existing prediction TSV as a label-free E-value')
    escore.add_argument('--folder_path', default='',
                        help='Data folder; used only to locate --output_dir when that is omitted')
    escore.add_argument('--output_dir', default='',
                        help='Folder holding window_predictions.jsonl and the prediction TSV')
    escore.add_argument('--file_name', default='assembly_prediction.tsv',
                        help='Input prediction TSV name inside --output_dir')
    escore.add_argument('--out_file_name', default='',
                        help='Output TSV name inside --output_dir '
                             '[default: assembly_prediction.escore.tsv]')
    escore.add_argument('--in_place', action='store_true',
                        help='Overwrite --file_name instead of writing a separate E-value file')
    escore.add_argument('--mode', default='calibrated', choices=list(ESCORE_MODES),
                        help='Empirical null used for the E-value')
    escore.add_argument('--r', default=ESCORE_DEFAULT_R, type=int,
                        help='Minimum reference-contig window count (calibrated mode only)')
    escore.add_argument('--floor', dest='floor', action='store_true', default=None,
                        help='Floor q [default: on for pooled, off for calibrated]')
    escore.add_argument('--no_floor', dest='floor', action='store_false',
                        help='Never floor q')
    escore.add_argument('--abstain_score', default=0.0, type=float,
                        help='Score given to contigs with no scored windows [default: 0]')
    escore.add_argument('--bp_min_e', default=E_P05, type=float,
                        help='Minimum E-value for writing a BreakPoint; below it the column '
                             f'stays -1 [default: {E_P05}]. Keep it at or below the '
                             '--threshold you intend to pass to `correct`')

    fine_tune = subparsers.add_parser('finetune', help='Fine-tune model')
    fine_tune.add_argument('--epochs', type=int, default = 50)
    fine_tune.add_argument('--batch_size', type=int, default= 256)
    fine_tune.add_argument('--lr', type=float, default= 0.0001)
    fine_tune.add_argument('--gpu_device', type=str, default='cuda:0')
    fine_tune.add_argument('--clip', type=int, default=1)
    fine_tune.add_argument('--weight_decay', type=float, default=0.0001)
    fine_tune.add_argument('--checkPoint_path', type=str, default=f'./fine-tune-model-weight')
    fine_tune.add_argument('--pretrain_model_path', type=str, default=f'./pretrain-model-weight/checkpoint.pt')
    fine_tune.add_argument('--finetune_dataset_path', type=str, default=f'')
    fine_tune.add_argument(
        '--unfreeze',
        choices=('linear', 'layer4', 'layer3_layer4', 'all'),
        default='layer4',
        help='Which pretrained layers to update during supervised fine-tuning',
    )
    fine_tune.add_argument(
        '--balanced_sampler',
        action='store_true',
        help='Balance classes per epoch without deleting negative features',
    )
    fine_tune.add_argument(
        '--selection_metric',
        choices=('loss', 'auprc'),
        default='auprc',
        help='Validation metric used to select the best checkpoint',
    )

    pretrain_data = subparsers.add_parser('pretrain_data', help='Generate pretrain dataset')
    pretrain_data.add_argument('--data_folder', type=str, default = '')
    pretrain_data.add_argument('--feature_folder', type=str, default = '')
    pretrain_data.add_argument('--label', type=str, default='all_alignments_final-contigs.tsv')
    pretrain_data.add_argument('--assembly', type=str, default='final.contigs.fa')
    pretrain_data.add_argument('--alignment', type=str, default='sort.bam')
    pretrain_data.add_argument('--pretrain_dataset_path', type=str, default=f'')

    finetune_data = subparsers.add_parser('finetune_data', help='Generate finetune dataset')
    finetune_data.add_argument('--data_folder', type=str, default = '')
    finetune_data.add_argument('--feature_folder', type=str, default = '')
    finetune_data.add_argument('--label', type=str, default='all_alignments_final-contigs.tsv')
    finetune_data.add_argument('--assembly', type=str, default='final.contigs.fa')
    finetune_data.add_argument('--alignment', type=str, default='sort.bam')
    finetune_data.add_argument('--eval_only', action='store_true')
    finetune_data.add_argument(
        '--data_workers',
        type=int,
        default=8,
        help='Parallel community-level feature generation workers',
    )
    finetune_data.add_argument(
        '--community_split_manifest',
        type=str,
        default='',
        help=(
            'Optional TSV with split, sample_id and path columns. '
            'When provided, whole communities—not contigs—define train/eval.'
        ),
    )
    finetune_data.add_argument(
        '--balance_output',
        action='store_true',
        help='Destructively downsample generated features to equal classes',
    )

    if not options:
        parser.print_help(sys.stderr)
        sys.exit()

    return parser.parse_args(options)

def main():
    options = sys.argv[1:]
    args = get_opts(options)

    warnings.filterwarnings("ignore")
    logger = logging.getLogger('DeepMM')
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(sh)

    logger.info('Start DeepMM')

    if args.cmd =='finetune':
        logger.info('Processing: Fine-tuning a new model')
        finetune(args)
        logger.info("Finished")

    if args.cmd == 'predict':
        logger.info('Processing: Identify metagenomic misassemblies (inference only)')
        predict(args)
        logger.info("Finished")

    if args.cmd == 'correct':
        logger.info('Processing: Split misassembled contigs from prediction TSV')
        correct(args)
        logger.info("Finished")

    if args.cmd == 'escore':
        logger.info('Processing: Rewriting predictions as label-free E-values')
        out_dir = args.output_dir or os.path.join(args.folder_path, 'deepmm_output')
        if not os.path.isdir(out_dir):
            raise SystemExit(f'output dir not found: {out_dir}')
        out_name = (args.file_name if args.in_place
                    else (args.out_file_name or 'assembly_prediction.escore.tsv'))
        path, summary = run_escore(
            out_dir,
            out_tsv=os.path.join(out_dir, out_name),
            file_name=args.file_name,
            mode=args.mode,
            r=args.r,
            floor=args.floor,
            abstain_score=args.abstain_score,
            bp_min_e=args.bp_min_e,
        )
        logger.info(f'Wrote {path}')
        logger.info(describe_escore(summary))
        logger.info("Finished")

    if args.cmd == 'pretrain_data':
        logger.info('Processing: Generating pretrain data')
        get_pretrain_data(args)
        logger.info("Finished")

    if args.cmd == 'finetune_data':
        logger.info('Processing: Generating fintune data')
        get_fine_tune_dataset(args)
        logger.info("Finished")

if __name__ == '__main__':
    main()
