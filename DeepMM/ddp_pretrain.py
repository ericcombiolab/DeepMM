import os
import sys
import random
import argparse
import logging
import warnings
from datetime import timedelta

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import numpy as np
from tqdm import tqdm
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

from tools import *
from model import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class TrainingProcess():

    def __init__(self, model, args, local_rank):

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        self.args = args
        self.model = model
        self.local_rank = local_rank
        self.criterion = torch.nn.CrossEntropyLoss().cuda(self.local_rank)

    def training_figure(self, figure_path, x_epochs, train_emb_loss, train_pre_loss, eval_pre_loss, y_eval_acc):
        x_epochs = np.array(x_epochs)
        train_emb_loss = np.array(train_emb_loss)
        train_pre_loss = np.array(train_pre_loss)
        eval_pre_loss = np.array(eval_pre_loss)
        y_eval_acc = np.array(y_eval_acc)
        interp_func_emb = make_interp_spline(x_epochs, train_emb_loss)
        interp_func_train_pre = make_interp_spline(x_epochs, train_pre_loss)
        interp_func_eval_pre = make_interp_spline(x_epochs, eval_pre_loss)
        interp_func_eval_acc = make_interp_spline(x_epochs, y_eval_acc)

        x_smooth = np.linspace(x_epochs.min(), x_epochs.max(), 300)

        y_1_smooth = interp_func_emb(x_smooth)
        y_2_smooth = interp_func_train_pre(x_smooth)
        
        y_3_smooth = interp_func_eval_pre(x_smooth)
        y_4_smooth = interp_func_eval_acc(x_smooth)
        
        os.makedirs(figure_path, exist_ok=True)
        plt.xlabel("epochs")
        plt.plot(x_smooth, y_1_smooth, color = 'red', label = 'Embedding Loss')
        plt.plot(x_smooth, y_2_smooth, color = 'blue', label = 'Training Prediction Loss')
        plt.plot(x_smooth, y_3_smooth, color = 'green', label = 'Evaluating Prediction Loss')
        plt.plot(x_smooth, y_4_smooth, color = 'orange', label = 'Evaluating Accuracy Loss')
        
        plt.legend()
        plt.savefig(f'{figure_path}/figure.jpg')
        plt.close()
        logging.info(f'Traning figure saved at {figure_path}')

    def acc_figure(self, figure_path, x_epochs, y_eval_acc):
        x_epochs = np.array(x_epochs)
        y_eval_acc = np.array(y_eval_acc)
        interp_func_2 = make_interp_spline(x_epochs, y_eval_acc)

        x_smooth = np.linspace(x_epochs.min(), x_epochs.max(), 300)

        y_2_smooth = interp_func_2(x_smooth)
        os.makedirs(figure_path, exist_ok=True)
        plt.xlabel("epochs")
        plt.plot(x_smooth, y_2_smooth, color = 'blue', label = 'evaluate acc')
        plt.legend()
        plt.savefig(f'{figure_path}/megahit_vit_{self.epochs}_acc_figure.jpg')
        plt.close()
        logging.info(f'Accuracy figure saved at {figure_path}')

    def get_lr(self, optimizer: torch.optim):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def cal_metrics(self,dataset):
        
        dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = self.args.batch_size, shuffle = True, drop_last = True, num_workers=8)
        self.model.eval()
        true_label = []
        pred_label = []

        with torch.no_grad():
            for _, features, labels in tqdm(dataloader, desc = 'Calculating accuracy ...'):
                features = features.cuda(self.local_rank)
                _, outputs = self.model(features)
                outputs = outputs.cpu()
                true_label.extend(torch.argmax(labels, dim=1)) 
                pred_label.extend(torch.argmax(outputs, dim=1))

        acc = sm.accuracy_score(true_label, pred_label)
        return round(acc, 3)

    def info_nce_loss(self, features):
        """
        Calculate the InfoNCE loss.

        :param features: Input features.
        :return: Logits and labels for the loss.
        """
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda(self.local_rank)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda(self.local_rank)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)


        positives = similarity_matrix[labels.bool()].view(-1, 1)

        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        negatives = negatives[:, None].expand(-1, self.args.n_views - 1, -1).flatten(0, 1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(self.local_rank)

        logits = logits / self.args.temperature
        return logits, labels
    
    def train(self, training_dataset, evaluating_dataset):
        try:
            if self.local_rank == 0:
                os.makedirs(self.args.checkPoint_path, exist_ok=True)
                logging.basicConfig(filename=f'{self.args.checkPoint_path}/training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
                best_eval_loss = float('inf')
                x_epochs = []
                train_emb_loss = []
                train_pre_loss = []
                eval_pre_loss = []
                y_eval_acc = []
                logging.info('Training start ...')
            
            scaler = GradScaler(enabled=self.args.fp16_precision)
            optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr, weight_decay=self.args.weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.8)
            for epoch_counter in range(self.args.epochs):
                train_sampler = DistributedSampler(training_dataset)
                train_loader = torch.utils.data.DataLoader(  
                                                            dataset = training_dataset,
                                                            batch_size = self.args.batch_size, 
                                                            drop_last = True, 
                                                            num_workers=8,
                                                            sampler = train_sampler
                                                            )
                train_sampler.set_epoch(epoch_counter)

                self.model.train()
                emb_epoch_loss = 0
                pre_epoch_loss = 0
                batch_count = 0                
                
                if self.local_rank == 0:
                    lr = self.get_lr(optimizer)
                    train_loader = tqdm(train_loader,  desc = f'Epoch [{epoch_counter+1}/{self.args.epochs}], lr ({lr:1.8f})')
              
                #* Train
                for images, window_labels in train_loader:

                    optimizer.zero_grad()

                    batch_count += 1
                    window_labels = window_labels.cuda(self.local_rank)
                    multi_images = torch.cat(images, dim=0)
                    multi_images = multi_images.cuda(self.local_rank)

                    with autocast(enabled=self.args.fp16_precision, device_type='cuda'):
                        embeddings, outputs = self.model(multi_images)
                        outputs = outputs[:self.args.batch_size]
                        logits, labels = self.info_nce_loss(embeddings)
                        loss_1 = self.criterion(logits, labels)
                        loss_2 = self.criterion(outputs, window_labels)
                        emb_epoch_loss += loss_1.item()
                        pre_epoch_loss += loss_2.item()

                    loss =  loss_1 +  loss_2
                
                    scaler.scale(loss).backward()

                    scaler.step(optimizer)
                    scaler.update()
                
                dist.barrier()
                if self.local_rank == 0:
                    #* Eval
                    eval_dataloader = torch.utils.data.DataLoader(
                                                dataset = evaluating_dataset,
                                                batch_size = self.args.batch_size,
                                                shuffle = True,
                                                drop_last = True,
                                                num_workers=8
                                                )
                    self.model.eval()
                    eval_epoch_loss = 0
                    batch_count = 0
                    with torch.no_grad():
                        for _, features, labels in tqdm(eval_dataloader, desc='Evaluating ...'):
                            batch_count += 1
                            features, labels = features.cuda(self.local_rank), labels.cuda(self.local_rank)
                            _, outputs = self.model(features)
                            loss = self.criterion(outputs, labels)
                            eval_epoch_loss += loss.item()
                    evaluate_loss = eval_epoch_loss / batch_count 

                    eval_acc= self.cal_metrics(evaluating_dataset)
                    logging.info(f'Epoch [{epoch_counter+1}/{self.args.epochs}]')
                    logging.info(F'Embedding Loss :{emb_epoch_loss / batch_count}')
                    logging.info(F'Train Prediction Loss :{pre_epoch_loss / batch_count}')
                    logging.info(F'Eval Prediction Loss :{evaluate_loss}')
                    logging.info(F'Eval accuracy: {eval_acc}')
                    # Model save
                    if evaluate_loss < best_eval_loss:
                        best_eval_loss = evaluate_loss
                        torch.save(self.model.module.state_dict(), os.path.join(self.args.checkPoint_path, f'checkpoint.pt'))
                        logging.info('Model saved')

                    x_epochs.append(epoch_counter+1)
                    train_emb_loss.append(emb_epoch_loss / batch_count)
                    train_pre_loss.append(pre_epoch_loss / batch_count)
                    eval_pre_loss.append(evaluate_loss)
                    y_eval_acc.append(eval_acc)

                scheduler.step()

            torch.cuda.empty_cache()
            if self.local_rank == 0:
                self.training_figure(self.args.checkPoint_path, x_epochs, train_emb_loss, train_pre_loss, eval_pre_loss, y_eval_acc)

        except Exception as e:
            logging.error(f"Process {self.local_rank} encountered an error: {e}")

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Identify and correct metagenomic misassemblies with deep learning')
    parser.add_argument('--eval_dataset_path', metavar='DIR', default='', help='path to eavl dataset')
    parser.add_argument('--pretrain_dataset_path', metavar='DIR', default='', help='path to pretrain dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', help='model architecture: resnet18 | resnet 50 (default: resnet50)')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N', help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--seed', default=42, type=int, help='seed for initializing training.')
    parser.add_argument('--out_dim', default=128, type=int, help='feature dimension (default: 128)')
    parser.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')
    parser.add_argument('--n_views', default=5, type=int, metavar='N', help='Number of views for contrastive learning training.')
    parser.add_argument('--gpus', default=1, type=int, help='Number of GPUs.')
    parser.add_argument('--checkPoint_path', type=str, default=f'./pretiran-model-weight', help='Checkpoint path')
    parser.add_argument('--fp16-precision', action='store_true', help='Whether or not to use 16-bit precision GPU training.')
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        warnings.filterwarnings("ignore")
        logger = logging.getLogger('DeepMM')
        logger.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(sh)

        logger.info('Start DeepMM')
        

    pretrain_dataset_path = args.pretrain_dataset_path
    eval_dataset_path = args.eval_dataset_path
    
    if local_rank == 0:
        logger.info('Processing: Loading Dataset ...')

    training_dataset = Pretrain_FeatureDataset(pretrain_dataset_path)
    evaluating_dataset = EvalDataset(eval_dataset_path, 'eval')

    if local_rank == 0:
        logger.info('Processing: Loading Model ...')

    model = LinearEvaluation(ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim))
    torch.distributed.init_process_group('nccl', world_size = args.gpus, rank = local_rank, timeout = timedelta(minutes=300))
    torch.cuda.set_device(local_rank)
    model = DistributedDataParallel(model.cuda(local_rank), device_ids = [local_rank])

    if local_rank == 0:
        logger.info('Processing: Pretraining a new model ...')

    tp = TrainingProcess(model=model, args=args, local_rank=local_rank)
    tp.train(training_dataset, evaluating_dataset)

    if local_rank == 0:
        logger.info("Finished")

if __name__ == '__main__':
    main()
