from _0_config_fc import seed_everything, device

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


def extract_feature(model, data_loader, finetuned_path):
    model.load_state_dict(torch.load(finetuned_path + '/best_model.pt'))
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()  # 평가 모드로 설정
    feature_extractor = feature_extractor.to(device)
    
    features_list = []
    paths_list = []
    labels_list = []
    
    with torch.no_grad():
        for path, images, labels in tqdm(iter(data_loader)):
            images = images.to(device)

            # ResNet 특징 추출 (마지막 풀링 층까지)
            features = feature_extractor(images)

            # [16, 512, 1, 1] 차원에서 [16, 512]로 변환
            features = features.squeeze(-1).squeeze(-1)

            # CPU로 이동 및 저장
            features_list.append(features.cpu())
            paths_list.extend(path)
            labels_list.extend(labels.numpy())

    features = torch.cat(features_list, dim=0)
    
    return features, paths_list, labels_list    


def vis_history_enhanced(history, num_classes, num_epochs, model_ckpt):
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
        plt.savefig(f'{model_ckpt}/history_mm.png')
        plt.show()
        plt.close()


# 새로운 함수: 특징 벡터 확장 기능 분리
def enhance_features(features, paths_list, df, n_quanti):  # 🔹 paths_list 추가
    base_v10 = ['alpha_p2', 'alpha_p1', 'alpha_0', 'alpha_m1', 'alpha_m2', 'alpha_m3', 
                'alpha_m4', 'alpha_m5', 'alpha_m6', 'alpha_m7']
    quanti_map = {
        10: base_v10,
        10.5: base_v10 + ['distance'],
        11: base_v10 + ['count'],
        11.5: base_v10 + ['count', 'distance'],
        12: base_v10 + ['alpha_33', 'alpha_67'],
        12.5: base_v10 + ['alpha_33', 'alpha_67', 'distance'],
        13: base_v10 + ['alpha_33', 'alpha_67', 'count'],
        13.5: base_v10 + ['alpha_33', 'alpha_67', 'count', 'distance'],
        8: ['alpha_p1', 'alpha_0', 'alpha_m1', 'alpha_m2', 'alpha_33', 'alpha_67', 'count', 'distance'],
        6: ['alpha_p1', 'alpha_0', 'alpha_m1', 'alpha_m2', 'alpha_33', 'alpha_67'],
        5: ['alpha_p1', 'alpha_0', 'alpha_m1', 'alpha_m2', 'alpha_33']
    }
    quanti_columns = quanti_map.get(n_quanti)
    
    print(f"\n=== enhance_features Debug ===")
    print(f"Features length: {len(features)}")
    print(f"Paths_list length: {len(paths_list)}")
    print(f"DF length: {len(df)}")
    print(f"First path from paths_list: {paths_list[0]}")
    print(f"First path from df: {df['img_dir'].iloc[0]}")
    
    # 첫 번째 매칭 확인
    row = df[df['img_dir'] == paths_list[0]]
    print(f"Match found: {len(row)} rows")
    if len(row) > 0:
        print(f"First alpha_p2: {row.iloc[0]['alpha_p2']}")
    
    
    enhanced_features = []
    for i, (feature, path) in enumerate(zip(features, paths_list)):
        row = df[df['img_dir'] == path].iloc[0]
        alpha_values = [row[col] for col in quanti_columns]
        enhanced_feature = np.append(feature, alpha_values)
        enhanced_features.append(enhanced_feature)
    
    return np.array(enhanced_features, dtype=np.float32)  # 🔹 dtype 추가

def calculate_class_weight(train_loader, num_classes=2):
    class_counts = [0] * num_classes
    total_samples = 0
    
    for _, _, labels in train_loader:
        labels = labels.view(-1)
        
        for label in labels:
            class_counts[label.item()] += 1
            total_samples += 1

    weights = [1 - (count / total_samples) for count in class_counts]
    weights_ts = torch.FloatTensor(weights)
    
    return weights, weights_ts

def train_enhanced_features(fc_layer_enhanced, n_quanti,
                            train_loader, train_features, train_paths_list, train_labels_list, train_df, 
                            val_features, val_paths_list, val_labels_list, val_df,
                            num_epochs, batch_size, optimizer, scheduler, label_smoothing,
                            model_ckpt, device, seed = 42):

    seed_everything(seed) 
    # 학습 및 검증 데이터 준비
    train_labels = np.array(train_labels_list)
    val_labels = np.array(val_labels_list)
    
    # 특징 벡터 확장
    enhanced_train_features = enhance_features(train_features, train_paths_list, train_df, n_quanti)  
    enhanced_val_features = enhance_features(val_features, val_paths_list, val_df, n_quanti)  
    
    train_paths = train_df['img_dir'].values  # 학습 이미지 경로
    val_paths = val_df['img_dir'].values  # 검증 이미지 경로
    
    cl_list, class_weight = calculate_class_weight(train_loader, num_classes=2)
    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=label_smoothing).to(device)
    
    weight_dict = {i : cl_list[i] for i in range(len(cl_list))}
    json_path = os.path.join(model_ckpt, 'class_weights_mm.json')
    with open(json_path, 'w') as f:
        json.dump(weight_dict, f, indent=4)    
        
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
    
    epoch_models_dir = os.path.join(model_ckpt, 'epoch_models')
    if not os.path.exists(epoch_models_dir):
        os.makedirs(epoch_models_dir)
        
    # 학습 루프
    for epoch in range(1, num_epochs + 1):
        fc_layer_enhanced.train()
        train_loss = []
        probs, preds, trues, paths = [], [], [], []

        for i in tqdm(range(0, len(enhanced_train_features), batch_size)):
            # 배치 데이터 준비 (인덱스 슬라이싱 사용)
            end_idx = min(i+batch_size, len(enhanced_train_features))
            batch_X = torch.tensor(enhanced_train_features[i:end_idx], dtype=torch.float32).to(device)
            batch_y = torch.tensor(train_labels[i:end_idx], dtype=torch.long).to(device)
            batch_paths = [train_paths[idx] for idx in range(i, end_idx)]

            if epoch == 1 and i == 0:  # 🔹 첫 epoch 첫 배치
                # 🔹 여기에 추가!
                fc_layer_enhanced.eval()
                with torch.no_grad():
                    test_out = fc_layer_enhanced(batch_X)
                print(f"[EVAL] Real batch output sum: {test_out.sum().item():.15f}")
                fc_layer_enhanced.train()  # 다시 train mode            
            
            # 순전파, 역전파, 가중치 업데이트
            optimizer.zero_grad()
            output = fc_layer_enhanced(batch_X)

            probability = torch.softmax(output, dim=1)[:, 1]
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
        _val_loss, _val_auc, _val_acc, _val_sensitivity, _val_precision, _val_specificity, _val_f1, _val_result = validation_enhanced(
            fc_layer_enhanced, criterion, enhanced_val_features, val_labels, val_paths, device, model_ckpt, batch_size
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
            
        # 모든 에포크마다 모델 저장
      #  epoch_model_path = os.path.join(epoch_models_dir, f'model_epoch_{epoch}.pt')
      #  torch.save(fc_layer_enhanced.state_dict(), epoch_model_path)
        
        # 각 에포크의 성능 지표도 저장
        epoch_metric = {
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
        with open(os.path.join(epoch_models_dir, f'metrics_epoch_{epoch}.json'), 'w') as f:
            json.dump(epoch_metric, f, indent=4)
            
        
        # 최고 성능 모델 저장
        if best_val_f1 < _val_f1:
            best_val_f1 = _val_f1
            best_model = fc_layer_enhanced
            best_epoch = epoch

            # 최고 성능 모델을 즉시 저장 (이 위치로 이동)
            torch.save(fc_layer_enhanced.state_dict(), model_ckpt + '/best_model_mm.pt')

            # 최고 성능 모델의 지표를 즉시 저장 (이 위치로 이동)
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
            with open(model_ckpt + '/best_model_metric_mm.json', 'w') as f:
                json.dump(best_model_metric, f, indent=4)

            # best model일 때 train 결과 저장
            train_result = pd.DataFrame({
                'image_path': paths,
                'prob': probs,
                'pred': preds,
                'label': trues
            })
            train_result.image_path = [p.split('/')[-1] for p in train_result.image_path]
            train_result.to_csv(model_ckpt + '/train_result_mm.csv', index=False, sep=',')
            _val_result.to_csv(model_ckpt + '/val_result_mm.csv', index=False, sep=',')
        best_val_metric = best_val_f1

        # 학습 이력 저장 및 시각화
        df_history = pd.DataFrame(history)
        df_history.to_csv(f'{model_ckpt}/history_mm.csv', sep=',', index=False)
        vis_history_enhanced(df_history, 2, num_epochs, model_ckpt)


    return best_model, best_val_metric


def validation_enhanced(fc_layer_enhanced, criterion, val_features, val_labels, val_paths, device, model_ckpt, batch_size):
    fc_layer_enhanced.eval()
    val_loss = []
    probs, preds, trues, paths = [], [], [], []
    
    with torch.no_grad():
        for i in range(0, len(val_features), batch_size):
            batch_X = torch.tensor(val_features[i:min(i+batch_size, len(val_features))], dtype=torch.float32).to(device)
            batch_y = torch.tensor(val_labels[i:min(i+batch_size, len(val_labels))], dtype=torch.long).to(device)
            batch_paths = val_paths[i:min(i+batch_size, len(val_paths))]
            
            logit = fc_layer_enhanced(batch_X)
            
            probability = torch.softmax(logit, dim=1)[:, 1]
            loss = criterion(logit, batch_y)    
            val_loss.append(loss.item())
            
            probs += probability.detach().cpu().numpy().tolist()
            preds += logit.argmax(dim=1).detach().cpu().numpy().tolist()
            trues += batch_y.detach().cpu().numpy().tolist()
            paths.extend(batch_paths)
            
    _val_loss = np.mean(val_loss)
    _val_auc = roc_auc_score(trues, probs)
    _val_acc = accuracy_score(trues, preds)
    _val_sensitivity = recall_score(trues, preds)
    _val_precision = precision_score(trues, preds)
    _val_specificity = recall_score(trues, preds, pos_label=0) 
    _val_f1 = f1_score(trues, preds)

    val_result = pd.DataFrame({'image_path': paths, 'prob': probs, 'pred': preds, 'label': trues})
    val_result.image_path = [p.split('/')[-1] for p in val_result.image_path]
  #  val_result.to_csv(model_ckpt + '/val_result_enhanced.csv', index=False, sep=',')
    
    return _val_loss, _val_auc, _val_acc, _val_sensitivity, _val_precision, _val_specificity, _val_f1, val_result


def inference_enhanced(fc_layer_enhanced, n_quanti, test_features, test_paths_list, test_df, device, model_ckpt, num_classes, batch_size, mode='test'):
    """
    ResNet18 특징 벡터와 추가 특징(alpha BR)을 사용하여 테스트 세트에서 추론 수행
    - validation_enhanced와 동일한 방식으로 처리하여 일관성 유지
    """
    # 모델 로드
  #  fc_layer_enhanced = nn.Linear(512 + 10, 2).to(device)
    fc_layer_enhanced.load_state_dict(torch.load(model_ckpt + '/best_model_mm.pt'))
    fc_layer_enhanced.to(device)
    fc_layer_enhanced.eval()
    
    # 테스트 데이터 준비
    test_labels = np.array(test_df['label'] if 'label' in test_df.columns else test_df.label)
    test_paths = test_df['img_dir'].values
    
    # 테스트 특징 벡터 확장 - validation과 동일한 방식으로 처리
    enhanced_test_features = enhance_features(test_features, test_paths_list, test_df, n_quanti)
    
    # 추론 수행
    probs, preds, trues, paths = [], [], [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(enhanced_test_features), batch_size)):
            # 배치 데이터 준비
            end_idx = min(i+batch_size, len(enhanced_test_features))
            batch_X = torch.tensor(enhanced_test_features[i:end_idx], dtype=torch.float32).to(device)
            batch_y = torch.tensor(test_labels[i:end_idx], dtype=torch.long).to(device)
            batch_paths = [test_paths[idx] for idx in range(i, end_idx)]
            
            # 추론
            logit = fc_layer_enhanced(batch_X)
            # 확률 및 예측값 계산
            probability = torch.softmax(logit, dim=1)[:, 1]
                
            # 결과 저장
            probs.extend(probability.detach().cpu().numpy().tolist())
            preds.extend(logit.argmax(dim=1).detach().cpu().numpy().tolist())
            trues.extend(batch_y.detach().cpu().numpy().tolist())
            paths.extend(batch_paths)
    
    # 성능 지표 계산
    _test_auc = roc_auc_score(trues, probs)
    _test_acc = accuracy_score(trues, preds)
    print(f'Test AUC : [{_test_auc:.3f}], Test acc : [{_test_acc:.3f}]')
    
    # 결과 저장
    results_df = pd.DataFrame({'image_path': paths, 'prob': probs, 'pred': preds, 'label': trues})
    results_df.image_path = [p.split('/')[-1] for p in results_df.image_path]
    results_df.to_csv(model_ckpt + f'/{mode}_inference_mm.csv', index=False, sep=',')
    
    # 테스트 지표 저장
    testset_metric = {
        "test_auc": _test_auc,
        "test_acc": _test_acc
    }
    
    # 테스트 결과 시각화
    metrics = vis_test_inference_enhanced(results_df, num_classes, model_ckpt, mode)
    
    # 모든 지표 저장
    with open(model_ckpt + f'/{mode}_metric_mm.json', 'w') as f:
        json.dump(testset_metric, f, indent=4)
    
    return _test_auc, results_df


def vis_test_inference_enhanced(test, num_classes, model_ckpt, mode='test'):
    """
    테스트 결과를 시각화하는 함수
    test.py의 vis_test_inference 함수와 유사
    """
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
        ax.set_title('metrics of test dataset', fontsize = 15)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        # 각 막대 위에 값 표시
        for i, (metric, value) in enumerate(metrics.items()):
            ax.text(i, value + 0.0005, str(value), ha='center', va='bottom', fontsize=20)
        plt.tight_layout()
        plt.savefig(model_ckpt + f'/{mode}_metrics_bar_mm.png', bbox_inches='tight', pad_inches=0.3)
        plt.show()
        plt.close()
              
    # 혼동 행렬 시각화
    cm = confusion_matrix(trues, preds)
    fig, ax = plt.subplots(figsize=(6, 6))
    
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
    
    plt.savefig(model_ckpt + f'/{mode}_confusion_mat_mm.png', bbox_inches='tight', pad_inches=0.3)
    plt.show()   
    plt.close()
    
    return metrics

def roc_plot_mm(model_ckpt):
    train = pd.read_csv(model_ckpt + '/train_result_mm.csv')
    val = pd.read_csv(model_ckpt + '/val_result_mm.csv')
    test = pd.read_csv(model_ckpt + '/test_inference_mm.csv')
    
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
    plt.savefig(model_ckpt + '/roc_curves_mm.png', bbox_inches='tight')
    plt.show()
    plt.close()
    