from _0_config_only_quanti import CFG, seed_everything, seed_worker, device
from _1_dataset import show_image_from_dataset, CustomDataset
from _2_model import get_model
from _2_model_fc import create_fc_model
from _3_train_only_quanti import train_quantitative_only, validation_quantitative_only, inference_quantitative_only, roc_plot, vis_history_quanti_only, vis_test_inference_quanti_only
from _4_test import inference
#from _3_train_fc_re3 import inference_enhanced, train_enhanced_features, extract_feature, roc_plot_mm

import os
import pandas as pd
from dfply import *
import argparse
import json
from collections import Counter
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from itertools import product

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
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--label_smoothing', type=float, default=0.08)
    parser.add_argument('--class_weight', type=str, default=True)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_ckpt', type=str, default=CFG['MODEL_CKPT'])
    
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--first_hidden', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=8)
    
    parser.add_argument('--use_dropout', type=str, default=True)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    
    parser.add_argument('--n_quanti', type=float, default=10)
    return parser.parse_args()

def objective(trial):
    trial_number = trial.number
    fold_list = [1,2,3,4,5]
    fold = fold_list[trial_number]       
    model_ckpt_dir = f'/storage01/user/choiheeji/ultrasound/mmode_output/paper_2025_01_Revision3/2_fo_final/cv/fold{fold}'
    CFG['EPOCHS'] = 50
    CFG['LEARNING_RATE'] = 1e-3
    CFG['BATCH_SIZE'] = 8
    CFG['WEIGHT_DECAY'] = 1e-4
    CFG['LABEL_SMOOTHING'] = 0
    CFG['SCHEDULER'] = False    
    CFG['CLASS_WEIGHT'] = True    
        
    CFG['SEED'] = 42
    CFG['MODEL_CKPT'] = model_ckpt_dir
    CFG['MODEL_NAME'] = 'resnet18' 
    CFG['NUM_CLASSES'] = 2
    CFG['N_QUANTI'] = 10 
    
    CFG['N_LAYERS'] = 2
    CFG['USE_DROPOUT'] = True     
    CFG['DROPOUT_RATE'] = 0.2
    CFG['FIRST_HIDDEN'] = 16
    CFG['HIDDEN_SIZE'] = None

    if  CFG['HIDDEN_SIZE'] is None:
        hidden_sizes = []
        
    else:
        if CFG['HIDDEN_SIZE'] > CFG['FIRST_HIDDEN']:
            CFG['HIDDEN_SIZE'] = CFG['FIRST_HIDDEN'] // 2
        hidden_sizes = [CFG['HIDDEN_SIZE']]

        
    quanti_columns = ['alpha_p2', 'alpha_p1', 'alpha_0', 'alpha_m1', 'alpha_m2', 'alpha_m3', 
                      'alpha_m4', 'alpha_m5', 'alpha_m6', 'alpha_m7']
    
    if not os.path.exists(CFG['MODEL_CKPT']):
        os.makedirs(CFG['MODEL_CKPT'])

    seed_everything(CFG['SEED'])  
    
    my_path = '/storage01/user/choiheeji/ultrasound/mmode_output/paper_2025_01_Revision3/0_dataset_tr_val/'
    CFG['DATASET'] = 'final_dataset.csv'
    filelist = pd.read_csv(my_path + CFG['DATASET']) >> arrange(X.filename)
    train_df = filelist[filelist[f'fold{fold}'] == 'train'].reset_index(drop=True)
    val_df = filelist[filelist[f'fold{fold}'] == 'val'].reset_index(drop=True)
    test_df = filelist[filelist[f'fold{fold}'] == 'test'].reset_index(drop=True)
    
    with open(os.path.join(CFG['MODEL_CKPT'], 'config.json'), 'w') as f:
        json.dump(CFG, f, indent=4)
         
    # FC 모델 생성 - 입력 크기를 정량적 특성의 개수로 설정
    fc_model_quanti_only = create_fc_model(
        input_size=len(quanti_columns),  # 정량적 특성의 수
        n_layers=CFG['N_LAYERS'],
        first_hidden=CFG['FIRST_HIDDEN'],
        hidden_sizes=hidden_sizes,
        use_dropout=CFG['USE_DROPOUT'],
        dropout_rate=CFG['DROPOUT_RATE'],
    ).to(device)
    
    # 정량적 특성만으로 모델 훈련
    best_model_quanti, best_val_metric_quanti = train_quantitative_only(
        fc_model=fc_model_quanti_only,
        quanti_columns = quanti_columns,
        train_df=train_df,
        val_df=val_df,
        num_epochs=CFG['EPOCHS'],
        batch_size=CFG['BATCH_SIZE'],
        lr=CFG['LEARNING_RATE'],
        wd=CFG['WEIGHT_DECAY'],
        label_smoothing = CFG['LABEL_SMOOTHING'],
        weight_yn = CFG['CLASS_WEIGHT'],
        model_ckpt=CFG['MODEL_CKPT'],
        device=device,
        num_classes=CFG['NUM_CLASSES']
    )
    
    # 테스트 데이터에서 정량적 특성 추출
    X_test = test_df[quanti_columns].values.astype(np.float32)
    y_test = test_df['label'].values
    path_test = test_df['img_dir'].values
    
    # 정량적 특성만으로 테스트
    test_auc_quanti, test_results_quanti = inference_quantitative_only(
        fc_model=fc_model_quanti_only,
        X_test=X_test,
        y_test=y_test,
        path_test=path_test,
        device=device,
        model_ckpt=CFG['MODEL_CKPT'],
        batch_size=CFG['BATCH_SIZE'],
        num_classes=CFG['NUM_CLASSES']
    )
    
    roc_plot(CFG['MODEL_CKPT'])
    
    return test_auc_quanti


if __name__ == '__main__':
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=CFG['SEED']+1),
                                direction='maximize')
    study.optimize(objective, n_trials=5)
