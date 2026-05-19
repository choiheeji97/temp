from _0_config_fc import CFG, seed_everything, seed_worker, device
from _1_dataset import show_image_from_dataset, CustomDataset
from _2_model import get_model
from _2_model_fc import create_fc_model
from _3_train import train
from _4_test import inference
from _3_train_fc_re3 import inference_enhanced, train_enhanced_features, extract_feature, roc_plot_mm

import ast
import os
import pandas as pd
from dfply import *
import argparse
import json
from collections import Counter
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

import optuna
import uuid


def parse_args():
    parser = argparse.ArgumentParser(description='Training configuration')
    parser.add_argument('--epochs', type=int, default=CFG['EPOCHS'])
    parser.add_argument('--learning_rate', type=float, default=CFG['LEARNING_RATE'])
    parser.add_argument('--batch_size', type=int, default=CFG['BATCH_SIZE'])
    parser.add_argument('--weight_decay', type=float, default=CFG['WEIGHT_DECAY'])
    
    parser.add_argument('--seed', type=int, default=CFG['SEED'])
    parser.add_argument('--model_ckpt', type=str, default=CFG['MODEL_CKPT'])
    parser.add_argument('--finetuned_path', type=str, default=CFG['FINETUNED_PATH'])
    parser.add_argument('--model_name', type=str, default=CFG['MODEL_NAME'], help='Model architecture choice')
    
    parser.add_argument('--n_layers', type=int, default=CFG['N_LAYERS'])
    parser.add_argument('--first_hidden', type=int, default=CFG['FIRST_HIDDEN'])
    parser.add_argument('--hidden_size', type=int, default=CFG['HIDDEN_SIZE'])
    
    parser.add_argument('--use_dropout', type=str, default=CFG['USE_DROPOUT'])
    parser.add_argument('--dropout_rate', type=float, default=CFG['DROPOUT_RATE'])
    
    parser.add_argument('--n_quanti', type=float, default=CFG['N_QUANTI'])
    
    return parser.parse_args()

def objective(trial):
    trial_number = trial.number
    fold_list = [1,2,3,4,5]
    fold = fold_list[trial_number]
    model_ckpt_dir = f'/storage01/user/choiheeji/ultrasound/mmode_output/paper_2025_01_Revision3/3_mm_final/cv_params1/fold{fold}'
                    
    CFG['EPOCHS'] = 50
    CFG['LEARNING_RATE'] = 1e-5
    CFG['BATCH_SIZE'] = 8
    CFG['WEIGHT_DECAY'] = 0.001
    CFG['LABEL_SMOOTHING'] = 0.08
    CFG['SCHEDULER'] = False    
        
    CFG['SEED'] = 42
    CFG['MODEL_CKPT'] = model_ckpt_dir
    CFG['FINETUNED_PATH'] = f'/storage01/user/choiheeji/ultrasound/mmode_output/paper_2025_01_Revision3/1_io_final/cv/fold{fold}'
    CFG['MODEL_NAME'] = 'resnet18' 
    CFG['NUM_CLASSES'] = 2

    CFG['N_QUANTI'] = 10 
    
    CFG['N_LAYERS'] = 2
    CFG['USE_DROPOUT'] = True     
    CFG['SCALE'] = False

    CFG['FIRST_HIDDEN'] = 384
    CFG['HIDDEN_SIZE'] = [32]
    CFG['DROPOUT_RATE'] = 0.1

    quanti_columns = ['alpha_p2', 'alpha_p1', 'alpha_0', 'alpha_m1', 'alpha_m2', 'alpha_m3', 
                      'alpha_m4', 'alpha_m5', 'alpha_m6', 'alpha_m7']
    
    if not os.path.exists(CFG['MODEL_CKPT']):
        os.makedirs(CFG['MODEL_CKPT'])

    seed_everything(CFG['SEED'])   
    
    my_path = '/storage01/user/choiheeji/ultrasound/mmode_output/paper_2025_01_Revision3/0_dataset_tr_val/'
    CFG['DATASET'] = 'final_dataset'
    filelist = pd.read_csv(my_path + CFG['DATASET'] + '.csv') >> arrange(X.filename)
    train_df = filelist[filelist[f'fold{fold}'] == 'train'].reset_index(drop=True)
    val_df = filelist[filelist[f'fold{fold}'] == 'val'].reset_index(drop=True)
    test_df = filelist[filelist[f'fold{fold}'] == 'test'].reset_index(drop=True)
    
    
    CFG['MODEL_PT'] = True
    model, CFG['IMG_SIZE'] = get_model(model_name=CFG['MODEL_NAME'], num_classes=CFG['NUM_CLASSES'])
    model.load_state_dict(torch.load(CFG['FINETUNED_PATH'] + '/best_model.pt'))
    
    print(model.__class__.__name__) 
    print(CFG['IMG_SIZE'])
    
    train_dataset = CustomDataset(train_df['img_dir'].values, train_df['label'].values, CFG['IMG_SIZE'], 
                                  train = True, model_ckpt=CFG['MODEL_CKPT'])
    val_dataset = CustomDataset(val_df['img_dir'].values, val_df['label'].values, CFG['IMG_SIZE'],
                               train = False, model_ckpt=CFG['MODEL_CKPT'])
    
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

    """ test """
    test_dataset = CustomDataset(test_df['img_dir'].values, test_df['label'].values, CFG['IMG_SIZE'], 
                                 train = False, model_ckpt=CFG['MODEL_CKPT'])
    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], 
                             shuffle=False, num_workers=0,
                            worker_init_fn=seed_worker, 
                             generator=torch.Generator().manual_seed(CFG['SEED']))
    
    train_features, train_paths_list, train_labels_list = extract_feature(model, train_loader, CFG['FINETUNED_PATH'])
    val_features, val_paths_list, val_labels_list = extract_feature(model, val_loader, CFG['FINETUNED_PATH'])
    test_features, test_paths_list, test_labels_list = extract_feature(model, test_loader, CFG['FINETUNED_PATH'])
    
    seed_everything(CFG['SEED']) 
    
    fc_layer_enhanced = create_fc_model(
        input_size=512 + len(quanti_columns), 
        n_layers=CFG['N_LAYERS'],
        first_hidden=CFG['FIRST_HIDDEN'],
        hidden_sizes=CFG['HIDDEN_SIZE'],
        use_dropout=CFG['USE_DROPOUT'],
        dropout_rate=CFG['DROPOUT_RATE'],
    ).to(device)
    
    optimizer = optim.Adam(params=fc_layer_enhanced.parameters(), 
                           lr=CFG["LEARNING_RATE"], 
                           weight_decay=CFG['WEIGHT_DECAY']) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                     patience=5, threshold=0.001,
                                                     threshold_mode='abs', 
                                                     min_lr=CFG["LEARNING_RATE"])
    
    best_model_enhanced, best_val_metric_enhanced = train_enhanced_features(
        fc_layer_enhanced=fc_layer_enhanced,
        n_quanti = CFG['N_QUANTI'],
        train_loader = train_loader, 
        train_features = train_features, 
        train_paths_list = train_paths_list, 
        train_labels_list = train_labels_list, 
        train_df = train_df,
        val_features = val_features, 
        val_paths_list = val_paths_list, 
        val_labels_list = val_labels_list, 
        val_df = val_df,
        num_epochs = CFG['EPOCHS'], 
        batch_size = CFG['BATCH_SIZE'],
        optimizer = optimizer, 
        scheduler = scheduler,
        label_smoothing =  CFG['LABEL_SMOOTHING'],
        model_ckpt = CFG['MODEL_CKPT'], 
        device = device,
        seed = CFG['SEED']
    )
    
    test_auc_enhanced, test_results_enhanced = inference_enhanced(
        fc_layer_enhanced, CFG['N_QUANTI'], test_features=test_features, test_paths_list=test_paths_list, test_df=test_df,
        device=device, model_ckpt=CFG['MODEL_CKPT'], num_classes=CFG['NUM_CLASSES'], batch_size=CFG['BATCH_SIZE'],
        mode = 'test'
    )
    test_auc_enhanced, test_results_enhanced = inference_enhanced(
        fc_layer_enhanced, CFG['N_QUANTI'], test_features=val_features, test_paths_list=val_paths_list, test_df=val_df,
        device=device, model_ckpt=CFG['MODEL_CKPT'], num_classes=CFG['NUM_CLASSES'], batch_size=CFG['BATCH_SIZE'],
        mode = 'val'
    )
    roc_plot_mm(CFG['MODEL_CKPT'])
    
    return test_auc_enhanced

if __name__ == '__main__':
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=CFG['SEED']+1),
                                direction='maximize')
    study.optimize(objective, n_trials=5)