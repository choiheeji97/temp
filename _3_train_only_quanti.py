from _0_config import seed_everything, device
from _1_dataset import CustomDataset

import os
import numpy as np
import pandas as pd
import json
from collections import Counter
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim


import seaborn as sns
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score, precision_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,roc_curve, auc

from torch.utils.data import DataLoader


def calculate_class_weight_df(train_df, label_col='label', num_classes=2):

    # 클래스별 개수
    class_counts = (
        train_df[label_col]
        .value_counts()
        .sort_index()
        .reindex(range(num_classes), fill_value=0)
    )

    total_samples = class_counts.sum()

    # 기존 방식 유지
    weights = [
        1 - (class_counts[i] / total_samples)
        for i in range(num_classes)
    ]

    weights_ts = torch.FloatTensor(weights)

    return weights, weights_ts

def train_quantitative_only(fc_model, quanti_columns,
                            train_df, val_df, num_epochs, batch_size, lr, wd, label_smoothing, weight_yn,
                            model_ckpt, device, num_classes=2):
    """
    정량적 특성만 사용하여 FC 모델을 훈련하는 함수
    
    Args:
        fc_model (nn.Module): 훈련할 FC 모델
        train_df (DataFrame): 훈련 데이터셋 (정량적 특성 포함)
        val_df (DataFrame): 검증 데이터셋 (정량적 특성 포함)
        num_epochs (int): 에폭 수
        batch_size (int): 배치 크기
        lr (float): 학습률
        wd (float): 가중치 감소
        model_ckpt (str): 모델 체크포인트 저장 경로
        device (torch.device): 훈련 장치 (CPU/GPU)
        num_classes (int): 클래스 수
    
    Returns:
        best_model, best_val_metric: 최고 성능 모델과 해당 검증 지표
    """

    
    # 훈련 및 검증 데이터에서 정량적 특성과 레이블 추출
    X_train = train_df[quanti_columns].values.astype(np.float32)
    y_train = train_df['label'].values
    path_train = train_df['img_dir'].values
    
    X_val = val_df[quanti_columns].values.astype(np.float32)
    y_val = val_df['label'].values
    path_val = val_df['img_dir'].values
    
    if weight_yn:
        cl_list, class_weight = calculate_class_weight_df(train_df, num_classes=2)
        criterion = nn.CrossEntropyLoss(weight=class_weight.to(device), label_smoothing=label_smoothing).to(device)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    
    optimizer = optim.Adam(fc_model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, 
        threshold_mode='abs', min_lr=lr, verbose=True
    )
    
    # 학습 이력 저장용 딕셔너리 초기화
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
    
    best_val_f1 = 0
    best_model = None
    best_epoch = 0
    
    # 학습 루프
    for epoch in range(1, num_epochs + 1):
        fc_model.train()
        train_loss = []
        probs, preds, trues, paths = [], [], [], []

        for i in range(0, len(X_train), batch_size):
            # 배치 데이터 준비
            end_idx = min(i+batch_size, len(X_train))
            batch_X = torch.tensor(X_train[i:end_idx], dtype=torch.float32).to(device)
            batch_y = torch.tensor(y_train[i:end_idx], dtype=torch.long).to(device)
            batch_paths = [path_train[idx] for idx in range(i, end_idx)]
                
            # 순전파, 역전파, 가중치 업데이트
            optimizer.zero_grad()
            output = fc_model(batch_X)
            
            probability = torch.softmax(output, dim=1)[:, 1] if num_classes == 2 else torch.softmax(output, dim=1)
            loss = criterion(output, batch_y)    
            loss.backward()
            optimizer.step()
            
            # 결과 저장
            train_loss.append(loss.item())
            probs += probability.detach().cpu().numpy().tolist()
            preds += output.argmax(dim=1).detach().cpu().numpy().tolist()
            trues += batch_y.detach().cpu().numpy().tolist()
            paths.extend(batch_paths)
            
        # 학습 지표 계산
        _train_loss = np.mean(train_loss)
        _train_auc = roc_auc_score(trues, probs)
        _train_acc = accuracy_score(trues, preds)
        
        
        # 검증 수행
        _val_loss, _val_auc, _val_acc, _val_sensitivity, _val_precision, _val_specificity, _val_f1, _val_result = validation_quantitative_only(
            fc_model, criterion, X_val, y_val, path_val, device, model_ckpt, batch_size, num_classes
        )
        
        # 결과 출력
        print(f'Epoch [{epoch}] \n Train Loss : [{_train_loss:.3f}] Val Loss : [{_val_loss:.3f}] \n '
              f'Train AUC : [{_train_auc:.3f}] Val AUC : [{_val_auc:.3f}] \n '
              f'Val Sensitivity: [{_val_sensitivity:.3f}] Val Precision: [{_val_precision:.3f}] '
              f'Val Specificity: [{_val_specificity:.3f}] Val f1: [{_val_f1:.3f}]')
        
        # 학습 이력 업데이트
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
        
        # 스케줄러 업데이트
        if scheduler is not None:
            scheduler.step(_val_f1)
            
        # 최고 성능 모델 저장
        if best_val_f1 < _val_f1:
            best_val_f1 = _val_f1
            best_model = fc_model
            best_epoch = epoch
            
            # 모델 저장
            torch.save(fc_model.state_dict(), model_ckpt + '/best_model_quanti_only.pt')
            
            # 성능 지표 저장
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
            with open(model_ckpt + '/best_model_metric_quanti_only.json', 'w') as f:
                json.dump(best_model_metric, f, indent=4)
                
            # best model일 때 train 결과 저장
            train_result = pd.DataFrame({
                'image_path': paths,
                'prob': probs,
                'pred': preds,
                'label': trues
            })
            train_result.image_path = [p.split('/')[-1] for p in train_result.image_path]
            train_result.to_csv(model_ckpt + '/train_result_only_quanti.csv', index=False, sep=',')
            _val_result.to_csv(model_ckpt + '/val_result_only_quanti.csv', index=False, sep=',')            

        # 학습 이력 저장 및 시각화
        df_history = pd.DataFrame(history)
        df_history.to_csv(f'{model_ckpt}/history_quanti_only.csv', sep=',', index=False)
        vis_history_quanti_only(df_history, num_classes, num_epochs, model_ckpt)
    
    if best_val_f1 == 0:
        torch.save(fc_model.state_dict(), model_ckpt + '/best_model_quanti_only.pt')
        best_model_metric = {
            'epoch': num_epochs,
            'note': 'last epoch (no improvement observed)'
        }
        with open(model_ckpt + '/best_model_metric_quanti_only.json', 'w') as f:
            json.dump(best_model_metric, f, indent=4)
            
        best_model = fc_model    
        
        # 성능 지표 저장
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
        with open(model_ckpt + '/best_model_metric_quanti_only.json', 'w') as f:
            json.dump(best_model_metric, f, indent=4)
                
        # best model일 때 train 결과 저장
        train_result = pd.DataFrame({
                'image_path': paths,
                'prob': probs,
                'pred': preds,
                'label': trues
        })
        train_result.image_path = [p.split('/')[-1] for p in train_result.image_path]
        train_result.to_csv(model_ckpt + '/train_result_only_quanti.csv', index=False, sep=',')
        _val_result.to_csv(model_ckpt + '/val_result_only_quanti.csv', index=False, sep=',')            

        # 학습 이력 저장 및 시각화
        df_history = pd.DataFrame(history)
        df_history.to_csv(f'{model_ckpt}/history_quanti_only.csv', sep=',', index=False)
        vis_history_quanti_only(df_history, num_classes, num_epochs, model_ckpt)
        
    return best_model, best_val_f1


def validation_quantitative_only(fc_model, criterion, X_val, y_val, path_val, device, model_ckpt, batch_size, num_classes=2):
    """정량적 특성만 사용하는 검증 함수"""
    fc_model.eval()
    val_loss = []
    probs, preds, trues, paths = [], [], [], []
    
    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            # 배치 데이터 준비
            end_idx = min(i+batch_size, len(X_val))
            batch_X = torch.tensor(X_val[i:end_idx], dtype=torch.float32).to(device)
            batch_y = torch.tensor(y_val[i:end_idx], dtype=torch.long).to(device)
            batch_paths = [path_val[idx] for idx in range(i, end_idx)]
            
            logit = fc_model(batch_X)
            probability = torch.softmax(logit, dim=1)[:, 1] 
            
            loss = criterion(logit, batch_y)    
            val_loss.append(loss.item())
            
            probs += probability.detach().cpu().numpy().tolist()
            preds += logit.argmax(dim=1).detach().cpu().numpy().tolist()
            trues += batch_y.detach().cpu().numpy().tolist()
            paths.extend(batch_paths)
            
    # 성능 지표 계산
    _val_loss = np.mean(val_loss)
    _val_auc = roc_auc_score(trues, probs)
    _val_acc = accuracy_score(trues, preds)
    _val_sensitivity = recall_score(trues, preds)
    _val_precision = precision_score(trues, preds)
    _val_specificity = recall_score(trues, preds, pos_label=0) 
    _val_f1 = f1_score(trues, preds)
    
    # 검증 결과 저장
    val_result = pd.DataFrame({'image_path': paths, 'prob': probs, 'pred': preds, 'label': trues})
  #  val_result.to_csv(model_ckpt + '/val_result_quanti_only.csv', sep=',')
    
    return _val_loss, _val_auc, _val_acc, _val_sensitivity, _val_precision, _val_specificity, _val_f1, val_result


def inference_quantitative_only(fc_model, X_test, y_test, path_test, device, model_ckpt, batch_size, num_classes=2):
    """정량적 특성만 사용하는 추론 함수"""
    # 모델 로드
    fc_model.load_state_dict(torch.load(model_ckpt + '/best_model_quanti_only.pt'))
    fc_model.to(device)
    fc_model.eval()
    
    # 추론 수행
    probs, preds, trues, paths = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            # 배치 데이터 준비
            end_idx = min(i+batch_size, len(X_test))
            batch_X = torch.tensor(X_test[i:end_idx], dtype=torch.float32).to(device)
            batch_y = torch.tensor(y_test[i:end_idx], dtype=torch.long).to(device)
            batch_paths = [path_test[idx] for idx in range(i, end_idx)]
            
            logit = fc_model(batch_X)
            probability = torch.softmax(logit, dim=1)[:, 1]

            probs.extend(probability.detach().cpu().numpy().tolist())
            preds.extend(logit.argmax(dim=1).detach().cpu().numpy().tolist())
            trues.extend(batch_y.detach().cpu().numpy().tolist())
            paths.extend(batch_paths)
            
    # 성능 지표 계산
    if num_classes == 2:
        _test_auc = roc_auc_score(trues, probs)
        _test_acc = accuracy_score(trues, preds)
        print(f'Test AUC : [{_test_auc:.3f}], Test acc : [{_test_acc:.3f}]')
    else:
        _test_auc = roc_auc_score(trues, probs, multi_class="ovr", average="micro")
        _test_acc = accuracy_score(trues, preds)
        print(f'Test AUC : [{_test_auc:.3f}], Test acc : [{_test_acc:.3f}]')
    
    # 결과 저장
    results_df = pd.DataFrame({'image_path': paths, 'prob': probs, 'pred': preds, 'label': trues})
    results_df.to_csv(model_ckpt + '/test_inference_quanti_only.csv', sep=',')
    
    # 테스트 지표 저장
    testset_metric = {
        "test_auc": _test_auc,
        "test_acc": _test_acc
    }
    
    # 테스트 결과 시각화
    metrics = vis_test_inference_quanti_only(results_df, num_classes, model_ckpt)
    
    # 모든 지표 저장
    with open(model_ckpt + '/test_metric_quanti_only.json', 'w') as f:
        json.dump(testset_metric, f, indent=4)
    
    return _test_auc, results_df

def roc_plot(model_ckpt):
    train = pd.read_csv(model_ckpt + '/train_result_only_quanti.csv')
    val = pd.read_csv(model_ckpt + '/val_result_only_quanti.csv')
    test = pd.read_csv(model_ckpt + '/test_inference_quanti_only.csv')
    
    # Calculate ROC curves for both validation and test sets
    fpr_train, tpr_train, _ = roc_curve(train['label'], train['prob'])
    fpr_val, tpr_val, _ = roc_curve(val['label'], val['prob'])
    fpr_test, tpr_test, _ = roc_curve(test['label'], test['prob'])

    roc_auc_train = auc(fpr_train, tpr_train)
    roc_auc_val = auc(fpr_val, tpr_val)
    roc_auc_test = auc(fpr_test, tpr_test)

    # Create ROC curve plot
    plt.figure(figsize=(6,6))

    # Plot validation ROC curve
    plt.plot(fpr_train, tpr_train, 
             label=f'Train (AUC = {roc_auc_train:.3f})')

    # Plot test ROC curve
    plt.plot(fpr_val, tpr_val, 
             label=f'Val    (AUC = {roc_auc_val:.3f})')
    # Plot test ROC curve
    plt.plot(fpr_test, tpr_test, 
             label=f'Test  (AUC = {roc_auc_val:.3f})')
    
    # Add random classifier line
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.3,)
    # Customize plot
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title('ROC Curves', fontsize=15)
    plt.legend(loc='lower right', fontsize=15)
    plt.grid(True, alpha=0.3)

    # Set axis limits
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])

    plt.tight_layout()
    plt.savefig(model_ckpt + '/roc_curves_only_quanti.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    
def vis_history_quanti_only(history, num_classes, num_epochs, model_ckpt):
    """정량적 특성만 사용한 모델의 학습 이력 시각화 함수"""
    # _5_train_fc_re.py의 vis_history_enhanced 함수와 동일한 구현
    if num_classes == 2:     
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6,10))

        ax1.plot(history['epoch'], history['train_loss'], label='Train Loss')
        ax1.plot(history['epoch'], history['val_loss'], label='Val Loss')
        ax1.set_ylabel('Loss', fontsize = 14)
        ax1.legend(fontsize = 12)
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
        
        ax1.legend(fontsize=12, loc='upper right')
        ax2.legend(fontsize=12, loc='lower right')
        ax3.legend(fontsize=12, loc='lower right')
        ax4.legend(fontsize=12, loc='lower right')

        plt.tight_layout()
        plt.savefig(f'{model_ckpt}/history_quanti_only.png')
        plt.show()
        plt.close()


def vis_test_inference_quanti_only(test, num_classes, model_ckpt):
    """정량적 특성만 사용한 모델의 테스트 결과 시각화 함수"""
    # _5_train_fc_re.py의 vis_test_inference_enhanced 함수와 유사한 구현
    trues = test['label']
    probs = test['prob']
    preds = test['pred']

    # 지표 시각화 (바 그래프)
    if num_classes == 2:
        acc = round(accuracy_score(trues, preds), 3)
        auc = round(roc_auc_score(trues, probs), 3)
        sen = round(recall_score(trues, preds), 3)
        spe = round(recall_score(trues, preds, pos_label=0), 3)
        pre = round(precision_score(trues, preds), 3)
        f1 = round(f1_score(trues, preds), 3)

        metrics = {
            'Accuracy': acc,
            'AUC': auc,
            'Sensitivity': sen,
            'Specificity': spe,
            'Precision': pre,
            'F1-score': f1, 
        }

        # 막대그래프 
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(metrics.keys(), metrics.values())
        ax.set_ylim(0, 1.1)
        ax.set_xlabel('Metrics', fontsize=15)
        ax.set_ylabel('Values', fontsize=15)
        ax.set_title('Metrics of test dataset (quantitative features only)', fontsize = 15)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        # 각 막대 위에 값 표시
        for i, (metric, value) in enumerate(metrics.items()):
            ax.text(i, value + 0.0005, str(value), ha='center', va='bottom', fontsize=20)
        plt.tight_layout()
        plt.savefig(model_ckpt + '/test_metrics_bar_quanti_only.png', bbox_inches='tight', pad_inches=0.3)
        plt.show()
        plt.close()
    
    # 혼동 행렬 시각화
    cm = confusion_matrix(trues, preds)
    fig, ax = plt.subplots(figsize=(6, 6)) #5,5
    
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_display.plot(cmap='Blues', ax=ax)

    ax.set_xlabel('Predict', fontsize=15)
    ax.set_ylabel('Label', fontsize=15)
    ax.set_title('prediction of test dataset', fontsize = 15)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)

    for texts in cm_display.text_.ravel():
        texts.set_fontsize(30)  
        texts.set_fontweight('bold') 
    
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
    
    plt.savefig(model_ckpt + '/confusion_mat_quanti_only.png', bbox_inches='tight', pad_inches=0.3)
    plt.show()   
    plt.close()
    
    return metrics

