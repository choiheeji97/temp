from _0_config import CFG, seed_everything, seed_worker, device
from _1_dataset import show_image_from_dataset, CustomDataset
from _2_model import get_model
from _3_train_ls_split import train
from _4_test import inference, roc_plot

import os
import random
import pandas as pd
import numpy as np
from dfply import *
import argparse
import json
from collections import Counter
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

from sklearn.model_selection import train_test_split

from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
#from pytorchtools import EarlyStopping

import optuna
import uuid


""" args """
def parse_args():
    parser = argparse.ArgumentParser(description='Training configuration')
    parser.add_argument('--epochs', type=int, default=CFG['EPOCHS'])
    parser.add_argument('--learning_rate', type=float, default=CFG['LEARNING_RATE'])
    parser.add_argument('--batch_size', type=int, default=CFG['BATCH_SIZE'])
    parser.add_argument('--weight_decay', type=float, default=CFG['WEIGHT_DECAY'])
    parser.add_argument('--label_smoothing', type=float, default=0.08)

    parser.add_argument('--seed', type=int, default=CFG['SEED'])
    parser.add_argument('--model_ckpt', type=str, default=CFG['MODEL_CKPT'])
    parser.add_argument('--model_name', type=str, default=CFG['MODEL_NAME'])
    parser.add_argument('--model_pt', type=str, default=CFG['MODEL_PT'])    

    parser.add_argument('--ds_ver', type=str, default='final_dataset')    
    
    return parser.parse_args()


""" main """
def objective(trial):
    trial_number = trial.number                       
    fold_list = [1,2,3,4,5]
    fold = fold_list[trial_number]
    model_ckpt_dir = f'/storage01/user/choiheeji/ultrasound/mmode_output/paper_2025_01_Revision3/1_io_final/cv_params3_e500/fold{fold}'
    """ config """
    CFG = {}
    CFG['EPOCHS'] = 500
    CFG['LEARNING_RATE'] = 5e-5 
    CFG['BATCH_SIZE'] = 8
    CFG['WEIGHT_DECAY'] = 1e-3 
    CFG['LABEL_SMOOTHING'] = 0.12 
                        
    CFG['SEED'] = 42 
    CFG['MODEL_CKPT'] = model_ckpt_dir
    CFG['MODEL_NAME'] = 'resnet18'
    CFG['MODEL_PT'] = True 
    
    CFG['DS_VER'] = 'final_dataset' 
    CFG['NUM_CLASSES'] = 2
                        
    
    if not os.path.exists(CFG['MODEL_CKPT']):
        os.makedirs(CFG['MODEL_CKPT'])

    """ data preparation """
    seed_everything(CFG['SEED'])        
    
    my_path = '/storage01/user/choiheeji/ultrasound/mmode_output/paper_2025_01_Revision3/0_dataset_tr_val/'
    
    filelist = pd.read_csv(my_path + CFG['DS_VER'] + '.csv') >> arrange(X.filename)
    train_df = filelist[filelist[f'fold{fold}'] == 'train'].reset_index(drop=True)
    val_df = filelist[filelist[f'fold{fold}'] == 'val'].reset_index(drop=True)  
    test_df = filelist[filelist[f'fold{fold}'] == 'test'].reset_index(drop=True)  
    
    """ model """
    model, CFG['IMG_SIZE'] = get_model(model_name=CFG['MODEL_NAME'], 
                                       num_classes=CFG['NUM_CLASSES'], pt = CFG['MODEL_PT'])
    
    print(model.__class__.__name__) 
    print(CFG['IMG_SIZE'])
    
    """ dataloader """
       
    train_dataset = CustomDataset(train_df['img_dir'].values, train_df['label'].values, CFG['IMG_SIZE'], 
                                  train = True, model_ckpt=CFG['MODEL_CKPT'])
    val_dataset = CustomDataset(val_df['img_dir'].values, val_df['label'].values, CFG['IMG_SIZE'],
                               train = False, model_ckpt=CFG['MODEL_CKPT'])
    
    show_image_from_dataset(train_dataset, 0, CFG['MODEL_CKPT'])
    
    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], 
                              num_workers=0, shuffle=True, drop_last=True, 
                             worker_init_fn=seed_worker, 
                              generator=torch.Generator().manual_seed(CFG['SEED']))
        
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'],
                            num_workers=0, shuffle=False, 
                            worker_init_fn=seed_worker, 
                            generator=torch.Generator().manual_seed(CFG['SEED']))
      
    with open(os.path.join(CFG['MODEL_CKPT'], 'config.json'), 'w') as f:
        json.dump(CFG, f, indent=4)
        
    """ training """
    model.to(device)
    
    optimizer = optim.Adam(params=model.parameters(), lr=CFG["LEARNING_RATE"], weight_decay=CFG['WEIGHT_DECAY']) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, 
                                                     threshold_mode='abs', min_lr=CFG["LEARNING_RATE"])
    
    best_model, best_val_metric = train(model, CFG['EPOCHS'], optimizer, train_loader, val_loader, 
                                        scheduler, CFG['LABEL_SMOOTHING'], device, model_ckpt=CFG['MODEL_CKPT'], 
                                        num_classes=CFG['NUM_CLASSES'], seed = None)
    

    inference(best_model, val_loader, device, model_ckpt=CFG['MODEL_CKPT'], 
              num_classes=CFG['NUM_CLASSES'], mode = 'val_id_date')
    

    
    test_dataset =  CustomDataset(test_df['img_dir'].values, test_df['label'].values, CFG['IMG_SIZE'], 
                                 train = False, model_ckpt=CFG['MODEL_CKPT'])
    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0,
                            worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(CFG['SEED']))

    inference(best_model, test_loader, device, model_ckpt=CFG['MODEL_CKPT'], 
              num_classes=CFG['NUM_CLASSES'], mode = 'test_id_date')    
    
    return best_val_metric

if __name__ == '__main__':
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=CFG['SEED']+1),
                                direction='maximize')
    study.optimize(objective, n_trials=5)
