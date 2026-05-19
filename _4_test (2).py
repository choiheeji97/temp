from _0_config import device
from _1_dataset import CustomDataset
from _2_model import get_model

import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score, precision_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,roc_curve, auc

import torch
from torch.utils.data import DataLoader



""" test """
        
""" test result """        
def vis_test_infernce(test, num_classes, model_ckpt, mode):
    trues = test['label']
    probs = test['prob']
    preds = test['pred']

    """ metrics bar plot """
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
        plt.savefig(model_ckpt + f'/test_metrics_bar_{mode}.png', bbox_inches='tight', pad_inches=0.3)
        plt.show()
        plt.close()
        
    """ confusion matrix """   
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
    
    plt.savefig(model_ckpt + f'/confusion_mat_{mode}.png', bbox_inches='tight', pad_inches=0.3)
    plt.show()   
    plt.close()
    
    return metrics


def inference(model, test_loader, device, model_ckpt, num_classes, mode):
    model.load_state_dict(torch.load(model_ckpt + '/best_model.pt'))
    model.to(device)
    model.eval()
    
    probs, preds, trues, paths = [], [], [], []
    with torch.no_grad():
        for path, images, labels in tqdm(iter(test_loader)):
            images = images.to(device)
            labels = labels.to(device)
            
            logit = model(images)
            probability = torch.softmax(logit, dim=1)[:, 1]

            probs += probability.detach().cpu().numpy().tolist()
            preds += logit.argmax(dim=1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()
            paths.extend(path) 

    _test_auc = roc_auc_score(trues, probs)
    _test_acc = accuracy_score(trues, preds)
    print(f'Test AUC : [{_test_auc:.3f}], Test acc : [{_test_acc:.3f}]') 

    results_df = pd.DataFrame({'image_path': paths, 'prob': probs, 'pred': preds, 'label': trues})
    results_df.image_path = results_df.image_path.str.split('/').str[-1]
    results_df.to_csv(model_ckpt + f'/test_inference_best_model_{mode}.csv', index=False, sep=',')
    
    testset_metric = {
        "test_auc": _test_auc,
        "test_acc": _test_acc
    }
    vis_test_infernce(results_df, num_classes, model_ckpt, mode)
    
    with open(model_ckpt + f'/test_metric_{mode}.json', 'w') as f:
        json.dump(testset_metric, f, indent=4)
    
    return _test_auc


def evaluate_model(test_inference):
    cm = confusion_matrix(test_inference['label'], test_inference['pred'])
    if num_classes == 2:
        print("Confusion Matrix (Binary Classification):")
    else:
        print("Confusion Matrix (Multi-class Classification):")
    print(cm)
    return cm


def roc_plot(model_ckpt):
    train = pd.read_csv(model_ckpt + '/train_result_best_model.csv')
    val = pd.read_csv(model_ckpt + '/val_result_best_model.csv')
    test = pd.read_csv(model_ckpt + '/test_inference_best_model_first.csv')
    
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
    plt.plot(fpr_test, tpr_test, 
             label=f'Test  (AUC = {roc_auc_test:.3f})')

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
    plt.savefig(model_ckpt + '/roc_curves_best_model.png', bbox_inches='tight')
    plt.show()
    plt.close()
    