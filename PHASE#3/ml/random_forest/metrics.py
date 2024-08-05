import matplotlib.pyplot as plt
import os
import json
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, cohen_kappa_score, balanced_accuracy_score


def calculate_metrics(y_true, y_pred, y_prob, num_classes=3):
    
    cm = confusion_matrix(y_true, y_pred)

    # Calculate metrics for each class
    metrics = []
    sensitivities = []
    specificities = []
    
    for i in range(len(cm)):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - TP - FP - FN
        
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        
        metrics.append({
            'sensitivity': sensitivity,
            'specificity': specificity
        })
    
    # Calculate overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate ROC-AUC and F1 score for validation set
    if num_classes == 2:
        auc = roc_auc_score(y_true, y_prob[:, 1])
        f1 = f1_score(y_true, y_pred)
    else:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Calculate averaged sensitivity and specificity
    avg_sensitivity = sum(sensitivities) / len(sensitivities) if sensitivities else 0
    avg_specificity = sum(specificities) / len(specificities) if specificities else 0
    
    return accuracy, metrics, auc, f1, cm, avg_sensitivity, avg_specificity

def custom_scorer(y_true, y_pred):
    # If y_pred is probabilities, convert to class predictions
    if y_pred.ndim == 2:
        y_pred_class = np.argmax(y_pred, axis=1)
    else:
        y_pred_class = y_pred

    # Calculate metrics
    auc = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovo')
    f1 = f1_score(y_true, y_pred_class, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred_class)
    balanced_acc = balanced_accuracy_score(y_true, y_pred_class)
    
    return 0.3 * auc + 0.3 * f1 + 0.2 * kappa + 0.2 * balanced_acc


def calculate_double_dichotomy_auc(y_true, y_prob):
    """
    Calculate AUC scores for double dichotomy classification.
    
    Parameters:
    y_true (array-like): True labels (0: Few, 1: Many, 2: None)
    y_prob (array-like): Predicted probabilities for each class
    
    Returns:
    dict: AUC scores for Normal vs Abnormal and Few vs Many
    """
    # 1. Normal (None) vs Abnormal (Few + Many)
    y_binary = (y_true != 2).astype(int)  # 2 is 'None', so we invert it for Normal vs Abnormal
    auc_normal_vs_abnormal = roc_auc_score(y_binary, y_prob[:, :2].sum(axis=1))
    
    # 2. Few vs Many (for Abnormal cases only)
    abnormal_mask = y_true != 2
    if np.sum(abnormal_mask) > 0:  # Ensure there are abnormal cases
        y_few_vs_many = (y_true[abnormal_mask] == 1).astype(int)  # 1 is 'Many'
        epsilon = 1e-15 # To avoid division by zero
        prob_few_vs_many = y_prob[abnormal_mask][:, 1] / (y_prob[abnormal_mask][:, 0] + y_prob[abnormal_mask][:, 1] + epsilon)
        auc_few_vs_many = roc_auc_score(y_few_vs_many, prob_few_vs_many)
    else:
        auc_few_vs_many = np.nan
    
    return {
        'auc_normal_vs_abnormal': auc_normal_vs_abnormal,
        'auc_few_vs_many': auc_few_vs_many
    }

def plot_confusion_matrix(cm, class_names, epoch_num=0, model_name='model', fold_num=0, save_dir='figures'):
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{save_dir}/confusion_matrix_{model_name}_epoch_{epoch_num}_fold{fold_num}.png')
    plt.close()

def custom_log(metrics, model_name, log_dir='logs'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = f'{log_dir}/{model_name}_metrics_log.json'
    with open(log_file, 'a') as f:
        f.write(json.dumps(metrics) + '\n')