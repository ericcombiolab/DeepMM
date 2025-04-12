import sys
import argparse
import logging
import warnings

from datetime import datetime
now = datetime.now()
formatted_time = now.strftime("%Y_%m_%d_%H_%M")

from .dataset_finetune import *
from .dataset_pretrain import *
from .training_finetune import *
from .training_pretrain import *
from .correct import *


def get_opts(options):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Identify and correct metagenomic misassemblies with deep learning')

    subparsers = parser.add_subparsers(title='DeepMM subcommands',
                                       dest='cmd',
                                       metavar='')

    correct = subparsers.add_parser('correct', help='Identify and correct misassemblies')
    correct.add_argument('--folder_path', default='', help='Path to data folder')
    correct.add_argument('--model_weight_path', default='./pretrain-model-weight/checkPoint.pt', help='Path to model weight')
    correct.add_argument('--fine_tune', action='store_true', help='Use Fine-tune model')
    correct.add_argument('-t', '--threads', default=8, type=int, help='Maximum number of threads [default: 8]')
    correct.add_argument('--gpu_device', default='cuda:0', help='GPU device id')

    pretrain = subparsers.add_parser('pretrain', help='Pretrain model')
    pretrain.add_argument('--eval_dataset_path', metavar='DIR', default='', help='path to eavl dataset')
    pretrain.add_argument('--pretrain_dataset_path', metavar='DIR', default='', help='path to pretrain dataset')
    pretrain.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', help='model architecture: resnet18 | resnet 50 (default: resnet50)')
    pretrain.add_argument('-j', '--workers', default=32, type=int, metavar='N', help='number of data loading workers (default: 32)')
    pretrain.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    pretrain.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 64)')
    pretrain.add_argument('--lr', '--learning-rate', default=0.0003, type=float, metavar='LR', help='initial learning rate', dest='lr')
    pretrain.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    pretrain.add_argument('--seed', default=42, type=int, help='seed for initializing training.')
    pretrain.add_argument('--out_dim', default=128, type=int, help='feature dimension (default: 128)')
    pretrain.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')
    pretrain.add_argument('--n_views', default=5, type=int, metavar='N', help='Number of views for contrastive learning training.')
    pretrain.add_argument('--gpus', default=1, type=int, help='Number of GPUs.')
    pretrain.add_argument('--checkPoint_path', type=str, default=f'./pretiran-model-weight/checkPoint_{formatted_time}', help='Checkpoint path')
    pretrain.add_argument('--fp16-precision', action='store_true', help='Whether or not to use 16-bit precision GPU training.')
    
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
    finetune_data.add_argument('--eval_only', type=str, default=False)

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

    if args.cmd == 'pretrain':
        logger.info('Processing: Pretraining a new model')
        pretrain(args)
        logger.info("Finished")

    if args.cmd =='finetune':
        logger.info('Processing: Fine-tuning a new model')
        finetune(args)
        logger.info("Finished")

    if args.cmd == 'correct':
        logger.info('Processing: Identify metagenomic misaseemblies and Correction')
        correct(args)
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
