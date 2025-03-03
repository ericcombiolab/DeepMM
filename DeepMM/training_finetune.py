import os
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sklearn.metrics as sm
import logging

from .tools import * 
from .model import *

class Trainer():
    def __init__(self, model: nn.Module, args):

        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.device = args.gpu_device
        self.clip = args.clip
        self.weight_decay = args.weight_decay
        self.checkPoint_path =args.checkPoint_path

        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)
        torch.cuda.manual_seed_all(42)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = 3, gamma = 0.8)

    def get_lr(self, optimizer: torch.optim):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def training_figure(self, figure_path, x_epochs, y_train_loss, y_eval_loss, window_len):
        x_epochs = np.array(x_epochs)
        y_train_loss = np.array(y_train_loss)
        y_eval_loss = np.array(y_eval_loss)
        interp_func_1 = make_interp_spline(x_epochs, y_train_loss)
        interp_func_2 = make_interp_spline(x_epochs, y_eval_loss)

        x_smooth = np.linspace(x_epochs.min(), x_epochs.max(), 300)

        y_1_smooth = interp_func_1(x_smooth)
        y_2_smooth = interp_func_2(x_smooth)
        os.makedirs(figure_path, exist_ok=True)
        plt.xlabel("epochs")
        plt.plot(x_smooth, y_1_smooth, color = 'red', label = 'train loss')
        plt.plot(x_smooth, y_2_smooth, color = 'blue', label = 'evaluate loss')
        plt.legend()
        plt.savefig(f'{figure_path}/megahit_vit_{self.epochs}_loss_figure.jpg')
        logging.info(f'Traning figure saved at {figure_path}')

    def acc_figure(self, figure_path, x_epochs, y_train_acc, y_eval_acc, window_len):
        x_epochs = np.array(x_epochs)
        y_train_acc = np.array(y_train_acc)
        y_eval_acc = np.array(y_eval_acc)
        interp_func_1 = make_interp_spline(x_epochs, y_train_acc)
        interp_func_2 = make_interp_spline(x_epochs, y_eval_acc)

        x_smooth = np.linspace(x_epochs.min(), x_epochs.max(), 300)

        y_1_smooth = interp_func_1(x_smooth)
        y_2_smooth = interp_func_2(x_smooth)
        os.makedirs(figure_path, exist_ok=True)
        plt.xlabel("epochs")
        plt.plot(x_smooth, y_1_smooth, color = 'red', label = 'train acc')
        plt.plot(x_smooth, y_2_smooth, color = 'blue', label = 'evaluate acc')
        plt.legend()
        plt.savefig(f'{figure_path}/megahit_vit_{self.epochs}_acc_figure.jpg')
        logging.info(f'Accuracy figure saved at {figure_path}')



    def train_(self,
                training_dataset,
                epoch,
                lr):
        
        train_dataloader = DataLoader(  
                                    dataset = training_dataset,
                                    batch_size = self.batch_size, 
                                    drop_last = True, 
                                    num_workers=8,
                                    )
        self.model.train()
        epoch_loss = 0
        batch_count = 0
        train_dataloader = tqdm(train_dataloader, desc = f'epoch ({epoch+1}), lr ({lr:1.8f})')

        for _, features, labels in train_dataloader:
            self.optimizer.zero_grad()
            batch_count += 1
            features, labels = features.cuda(self.device), labels.cuda(self.device)
            
            emb, predictions = self.model(features)
            loss = self.criterion(predictions, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

            epoch_loss += loss.item()
        train_loss = epoch_loss / batch_count
        return train_loss

    def evaluate(self, 
                evaluating_dataset):
        eval_dataloader = DataLoader(dataset = evaluating_dataset, batch_size = self.batch_size, shuffle = True, drop_last = True, num_workers=8)
        self.model.eval()
        epoch_loss = 0
        batch_count = 0
        with torch.no_grad():
            for _, features, labels in eval_dataloader:
                batch_count += 1
                features, labels = features.cuda(self.device), labels.cuda(self.device)
                emb, predictions = self.model(features)
                loss = self.criterion(predictions, labels)
                epoch_loss += loss.item()

        evaluate_loss = epoch_loss / batch_count
       
        return evaluate_loss
    
    def cal_metrics(self,dataset):
        dataloader = DataLoader(dataset = dataset, batch_size = self.batch_size, shuffle = True, drop_last = True, num_workers=8)
        self.model.eval()
        true_label = []
        pred_label = []

        with torch.no_grad():
            for _, features, labels in dataloader:
                features = features.cuda(self.device)
                emb, predictions = self.model(features)
                predictions = predictions.cpu()
                true_label.extend(torch.argmax(labels, dim=1)) 
                pred_label.extend(torch.argmax(predictions, dim=1))

        acc = sm.accuracy_score(true_label, pred_label)
        return round(acc, 3)


    def train(self,
            training_dataset,
            evaluating_dataset):
        
        best_eval_loss = float('inf')
        x_epochs = []
        y_train_loss = []
        y_eval_loss = []
        y_train_acc = []
        y_eval_acc = []
        os.makedirs(self.checkPoint_path, exist_ok=True)
        logging.basicConfig(filename=f'{self.args.checkPoint_path}/fine-tuneing.log', level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s')

        for epoch in range(self.epochs):
            lr = self.get_lr(self.optimizer)
            train_loss = self.train_(training_dataset, epoch, lr)

            evaluate_loss  = self.evaluate(evaluating_dataset)
            train_acc = self.cal_metrics(training_dataset)
            eval_acc= self.cal_metrics(evaluating_dataset)
            y_train_acc.append(train_acc)
            y_eval_acc.append(eval_acc)
            logging.info('Window length: {}'.format(config['window_len']))
            logging.info(f'Training loss: {train_loss}, acc: {train_acc}')
            logging.info(f'Evaluating loss: {evaluate_loss}, acc: {eval_acc}')

            # Model save
            if evaluate_loss < best_eval_loss:
                best_eval_loss = evaluate_loss
                torch.save(self.model.state_dict(), os.path.join(self.checkPoint_path, f'fine_tuned_checkPoint_{self.epochs}.pt'))
                logging.info('Evaluate loss model saved')

            x_epochs.append(epoch+1)
            y_train_loss.append(train_loss)
            y_eval_loss.append(evaluate_loss)

            self.scheduler.step()

        torch.cuda.empty_cache()
        self.training_figure(self.checkPoint_path, x_epochs, y_train_loss, y_eval_loss, config['window_len'])
        plt.clf()
        self.acc_figure(self.checkPoint_path, x_epochs, y_train_acc, y_eval_acc, config['window_len'])
        plt.clf()

def finetune(args):
    fine_tune_dataset_path = args.fine_tune_dataset_path
    training_dataset = EvalDataset(fine_tune_dataset_path,'train')
    evaluating_dataset = EvalDataset(fine_tune_dataset_path, 'eval')

    model = LinearEvaluation(ResNetSimCLR(base_model='resnet50', out_dim=128))
    model_path = args.pretrain_model_path
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = FinetuneModel(model)
    model = model.cuda(args.gpu_device)

    trainer = Trainer(model = model, args=args)
    trainer.train(training_dataset, evaluating_dataset)
    
