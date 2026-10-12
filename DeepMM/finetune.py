import os
import random

import logging
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import sklearn.metrics as sm


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
        self.balanced_sampler = args.balanced_sampler
        self.selection_metric = args.selection_metric

        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)
        torch.cuda.manual_seed_all(42)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.model = model
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            trainable, lr=self.lr, weight_decay=self.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = 3, gamma = 0.8)
        logging.info(
            "Trainable parameters: %d / %d",
            sum(p.numel() for p in trainable),
            sum(p.numel() for p in self.model.parameters()),
        )

    def _training_sampler(self, dataset):
        if not self.balanced_sampler:
            return None
        labels = [
            int(name[:-4].rsplit('_', 1)[-1])
            for name in dataset.feature_files
        ]
        counts = np.bincount(labels, minlength=2)
        if np.any(counts == 0):
            raise ValueError(
                f"Balanced sampler requires both classes, got counts={counts}"
            )
        sample_weights = [1.0 / counts[label] for label in labels]
        return WeightedRandomSampler(
            sample_weights, num_samples=len(labels), replacement=True
        )

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
        
        sampler = self._training_sampler(training_dataset)
        train_dataloader = DataLoader(
            dataset=training_dataset,
            batch_size=self.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=True,
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
        eval_dataloader = DataLoader(
            dataset=evaluating_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
        )
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
    
    def cal_metrics(self, dataset):
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
        )
        self.model.eval()
        true_label = []
        pred_label = []
        positive_score = []

        with torch.no_grad():
            for _, features, labels in dataloader:
                features = features.cuda(self.device)
                _, logits = self.model(features)
                probabilities = torch.softmax(logits, dim=1).cpu()
                batch_true = torch.argmax(labels, dim=1).numpy()
                true_label.extend(batch_true)
                pred_label.extend(torch.argmax(probabilities, dim=1).numpy())
                positive_score.extend(probabilities[:, 1].numpy())

        result = {
            "accuracy": float(sm.accuracy_score(true_label, pred_label)),
            "auroc": float(sm.roc_auc_score(true_label, positive_score)),
            "auprc": float(sm.average_precision_score(
                true_label, positive_score
            )),
        }
        return result


    def train(self,
            training_dataset,
            evaluating_dataset):
        
        best_eval_loss = float('inf')
        best_eval_auprc = -float('inf')
        x_epochs = []
        y_train_loss = []
        y_eval_loss = []
        y_train_acc = []
        y_eval_acc = []
        os.makedirs(self.checkPoint_path, exist_ok=True)
        logging.basicConfig(filename=f'{self.checkPoint_path}/fine-tuneing.log', level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s')

        for epoch in range(self.epochs):
            lr = self.get_lr(self.optimizer)
            train_loss = self.train_(training_dataset, epoch, lr)

            evaluate_loss  = self.evaluate(evaluating_dataset)
            train_metrics = self.cal_metrics(training_dataset)
            eval_metrics = self.cal_metrics(evaluating_dataset)
            y_train_acc.append(train_metrics["accuracy"])
            y_eval_acc.append(eval_metrics["accuracy"])
            logging.info('Window length: {}'.format(config['window_len']))
            logging.info(
                f"Training loss: {train_loss}, metrics: {train_metrics}"
            )
            logging.info(
                f"Evaluating loss: {evaluate_loss}, metrics: {eval_metrics}"
            )

            # Model save
            improved = (
                eval_metrics["auprc"] > best_eval_auprc
                if self.selection_metric == "auprc"
                else evaluate_loss < best_eval_loss
            )
            if improved:
                best_eval_loss = evaluate_loss
                best_eval_auprc = eval_metrics["auprc"]
                torch.save(self.model.state_dict(), os.path.join(self.checkPoint_path, f'fine_tuned_checkPoint_{self.epochs}.pt'))
                logging.info(
                    "Best model saved: eval_loss=%f eval_auprc=%f",
                    evaluate_loss,
                    eval_metrics["auprc"],
                )

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
    finetune_dataset_path = args.finetune_dataset_path
    training_dataset = EvalDataset(finetune_dataset_path,'train')
    evaluating_dataset = EvalDataset(finetune_dataset_path, 'eval')

    model_path = args.pretrain_model_path
    pretrain_state = torch.load(model_path, weights_only=True)
    arch = infer_arch_from_state_dict(pretrain_state)
    print(f'Detected pretrained backbone: {arch} ({model_path})')
    model = LinearEvaluation(ResNetSimCLR(base_model=arch, out_dim=128))
    model.load_state_dict(pretrain_state)
    model = FinetuneModel(model, unfreeze=args.unfreeze)
    model = model.cuda(args.gpu_device)

    trainer = Trainer(model = model, args=args)
    trainer.train(training_dataset, evaluating_dataset)
    
