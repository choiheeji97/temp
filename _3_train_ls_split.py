from _0_config import seed_everything, device
from _1_dataset import CustomDataset

import os
import numpy as np
import pandas as pd
import json
from collections import Counter
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score, precision_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
#from pytorchtools import EarlyStopping



""" train """


""" vis history """

""" training state """
def vis_history(history, num_classes, num_epochs, model_ckpt, seed = None):
    if num_classes == 2:     
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6,10))

        ax1.plot(history['epoch'], history['train_loss'], label='Train Loss')
        ax1.plot(history['epoch'], history['val_loss'], label='Val Loss')
        ax1.set_ylabel('Loss', fontsize = 14)
        ax1.legend(fontsize = 12)#, loc='upper right')
        ax1.set_title('training state', fontsize = 15)
        ax1.grid(True)

        ax2.plot(history['epoch'], history['train_auc'], label='Train AUC')
        ax2.plot(history['epoch'], history['val_auc'], label='Val AUC')
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel('AUC', fontsize = 14)
        ax2.legend(fontsize = 12, loc='lower right')
        ax2.grid(True)
        
        ax3.plot(history['epoch'], history['train_acc'], label='Train Accuracy')
        ax3.plot(history['epoch'], history['val_acc'], label='Val Accuracy')
        ax3.set_ylim(0, 1.1)
        ax3.set_ylabel('Accuracy', fontsize = 14)
        ax3.legend(fontsize = 12, loc='lower right')
        ax3.grid(True)

        ax4.plot(history['epoch'], history['val_sensitivity'], label='Val Sensitivity')
        ax4.plot(history['epoch'], history['val_specificity'], label='Val Specificity')
        ax4.plot(history['epoch'], history['val_f1'], label='Val F1-score')
        ax4.set_xlabel('Epoch', fontsize = 14)
        ax4.set_ylabel('Val sen/spe/f1', fontsize = 14)
        ax4.set_ylim(0, 1.1)
        ax4.legend(fontsize = 12, loc='lower right')
        ax4.grid(True)
        
        ax1.legend(fontsize=12, loc='upper right') # bbox_to_anchor=(1.05, 1)
        ax2.legend(fontsize=12, loc='lower right')#, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.legend(fontsize=12, loc='lower right')#, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.legend(fontsize=12, loc='lower right')#, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(model_ckpt)

        plt.close()

def calculate_class_weight(train_loader, num_classes=2):
    class_counts = [0] * num_classes
    total_samples = 0
    
    for _,_, labels in train_loader:
        labels = labels.view(-1)
        
        for label in labels:
            class_counts[label.item()] += 1
            total_samples += 1

    weights = [1 - (count / total_samples) for count in class_counts]
    weights_ts = torch.FloatTensor(weights)
    
    return weights, weights_ts

def train(model, num_epochs, optimizer, train_loader, val_loader, scheduler, label_sm,
          device, model_ckpt, num_classes, seed=None):
    model.to(device)

    cl_list, class_weight = calculate_class_weight(train_loader, num_classes)#torch.FloatTensor([0.2,0.8])
   
    print('Weighted Cross Entropy Loss:', class_weight)
    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=label_sm).to(device)
    
    weight_dict = {i : cl_list[i] for i in range(len(cl_list))}
    json_path = os.path.join(model_ckpt, 'class_weights.json')
    with open(json_path, 'w') as f:
        json.dump(weight_dict, f, indent=4)    
        
    best_epoch = 0
    """ binary or multi """
    if num_classes == 2:
        best_val_f1 = 0
        best_model = None
        history = {                             
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'train_acc': [],
            'val_auc': [],
            'val_acc': [],
            'val_sensitivity': [],
            'val_precision': [],
            'val_specificity': [],
            'val_f1': []
        }
    
        for epoch in range(1, num_epochs + 1):
            model.train()
            train_loss = []
            probs, preds, trues, paths = [], [], [], []

            for path, images, labels in tqdm(iter(train_loader)):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                output = model(images)

                # model output googlenet&inception or other
                if (model.__class__.__name__ in ['GoogLeNet', 'Inception3']) and (model.aux_logits is True):
                    output = output[0]
                else:
                    output = output

                probability = torch.softmax(output, dim=1)[:, 1]
                loss = criterion(output, labels)    
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

                probs += probability.detach().cpu().numpy().tolist()
                preds += output.argmax(dim=1).detach().cpu().numpy().tolist()
                trues += labels.detach().cpu().numpy().tolist()
                paths.extend(path)

            _train_loss = np.mean(train_loss)

            _train_auc = roc_auc_score(trues, probs)
            _train_acc = accuracy_score(trues, preds)
            _val_loss, _val_auc, _val_acc, _val_sensitivity, _val_precision, _val_specificity, _val_f1, _val_result = validation_binary(model, criterion,val_loader, device, model_ckpt)
            print(f'Epoch [{epoch}] \n Train Loss : [{_train_loss:.3f}] Val Loss : [{_val_loss:.3f}] \n Train AUC : [{_train_auc:.3f}] Val AUC : [{_val_auc:.3f}] \n Val Sensitivity: [{_val_sensitivity:.3f}] Val Precision: [{_val_precision:.3f}] Val Specificity: [{_val_specificity:.3f}] Val f1: [{_val_f1:.3f}]')

            # train history   
            history['epoch'].append(epoch)
            history['train_loss'].append(_train_loss)
            history['val_loss'].append(_val_loss)

            history['train_auc'].append(_train_auc)
            history['val_auc'].append(_val_auc)

            history['train_acc'].append(_train_acc)
            history['val_acc'].append(_val_acc)
            
            history['val_sensitivity'].append(_val_sensitivity)
            history['val_precision'].append(_val_precision)
            history['val_specificity'].append(_val_specificity)
            history['val_f1'].append(_val_f1)

            # scheduler
            if scheduler is not None:
                scheduler.step(_val_f1)

            # best model
            if best_val_f1 < _val_f1:
                best_val_f1 = _val_f1
                best_model = model
                best_epoch = epoch
                
                # seed가 있으면 파일명에 포함
                if seed is not None:
                    model_path = f'{model_ckpt}/best_model_seed_{seed}.pt'
                    metric_path = f'{model_ckpt}/best_model_metric_seed_{seed}.json'
                    train_result_path = f'{model_ckpt}/train_result_seed_{seed}.csv'
                    val_result_path = f'{model_ckpt}/val_result_seed_{seed}.csv'
                else:
                    model_path = f'{model_ckpt}/best_model.pt'
                    metric_path = f'{model_ckpt}/best_model_metric.json'
                    train_result_path = f'{model_ckpt}/train_result_best_model.csv'
                    val_result_path = f'{model_ckpt}/val_result_best_model.csv'                

                torch.save(model.state_dict(), model_path) # '/best_model.pt')
                
                best_model_metric = {
                    'epoch': epoch,
                    'train_loss': _train_loss,
                    'val_loss': _val_loss,
                    'train_auc': _train_auc,
                    'val_auc': _val_auc,
                    'train_acc': _train_acc,
                    'val_acc': _val_acc,
                    'val_sensitivity': _val_sensitivity,
                    'val_precision': _val_precision,
                    'val_specificity': _val_specificity,
                    'val_f1': _val_f1
                }
                with open(metric_path, 'w') as f:
                    json.dump(best_model_metric, f, indent=4)

                # best model일 때 train 결과 저장
                train_result = pd.DataFrame({
                    'image_path': paths,
                    'prob': probs,
                    'pred': preds,
                    'label': trues
                })
                train_result.image_path = train_result.image_path.str.split('/').str[-1]
                train_result.to_csv(train_result_path, index=False, sep=',')
                _val_result.to_csv(val_result_path, index=False, sep=',')
                
            best_val_metric = best_val_f1
            
            # save history 
            df_history = pd.DataFrame(history)

            if seed is not None:
                history_path = f'{model_ckpt}/history_seed_{seed}.csv'
                history_plot_path = f'{model_ckpt}/history_seed_{seed}.png'
            else:
                history_path = f'{model_ckpt}/history.csv'
                history_plot_path = f'{model_ckpt}/history.png'
                
            df_history.to_csv(history_path, sep=',', index=False)
            vis_history(df_history, num_classes, num_epochs, history_plot_path, seed = seed)  
                   
    return best_model, best_val_metric

def validation_binary(model, criterion, val_loader, device, model_ckpt):
    model.eval()
    val_loss = []
    probs, preds, trues, paths = [], [], [], []
    
    with torch.no_grad():
        for path, images, labels in tqdm(iter(val_loader)):
            images = images.to(device)
            labels = labels.to(device)
            logit = model(images)
            
            probability = torch.softmax(logit, dim=1)[:, 1]
            loss = criterion(logit, labels)    
            val_loss.append(loss.item())
            
            probs += probability.detach().cpu().numpy().tolist()
            preds += logit.argmax(dim=1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()
            paths.extend(path) 
            
    _val_loss = np.mean(val_loss)
    _val_auc = roc_auc_score(trues, probs)
    _val_acc = accuracy_score(trues, preds)
    _val_sensitivity = recall_score(trues, preds)
    _val_precision = precision_score(trues, preds)
    _val_specificity = recall_score(trues, preds, pos_label=0) 
    _val_f1 = f1_score(trues, preds)

    val_result = pd.DataFrame({'image_path': paths, 'prob': probs, 'pred': preds, 'label': trues})
    val_result.image_path = val_result.image_path.str.split('/').str[-1]
    
    return _val_loss, _val_auc, _val_acc, _val_sensitivity, _val_precision, _val_specificity, _val_f1, val_result



