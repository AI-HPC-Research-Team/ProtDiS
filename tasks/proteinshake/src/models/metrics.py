import numpy as np
import torch
from sklearn import metrics
from scipy.stats import spearmanr
from scipy.special import expit
import matplotlib.pyplot as plt

def fmax(probs, labels):
    thresholds = np.linspace(0, 1, 21)
    f_max = 0.0

    for threshold in thresholds:
        precision = 0.0
        recall = 0.0
        precision_cnt = 0
        recall_cnt = 0
        for idx in range(probs.shape[0]):
            prob = probs[idx]
            label = labels[idx]
            pred = (prob > threshold).astype(np.int32)
            correct_sum = (label * pred).sum()
            pred_sum = pred.sum()
            label_sum = label.sum()
            if pred_sum > 0:
                precision += correct_sum/pred_sum
                precision_cnt += 1
            if label_sum > 0:
                recall += correct_sum/label_sum
            recall_cnt += 1
        if recall_cnt > 0:
            recall = recall / recall_cnt
        else:
            recall = 0
        if precision_cnt > 0:
            precision = precision / precision_cnt
        else:
            precision = 0
        f = (2.*precision*recall)/max(precision+recall, 1e-8)
        f_max = max(f, f_max)

    return f_max

def default_metrics(task_type: str):
    """
    Return default metrics for the given task type.
    """
    if task_type == 'classification':
        return 'accuracy'
    elif task_type == 'binary':
        return 'mcc'
    elif task_type == 'multilabel':
        return 'fmax'
    elif task_type == 'regression':
        return 'spearmanr'
    else:
        raise ValueError(f"Unknown task_type: {task_type}, must be one of ['classification', 'regression', 'multilabel', 'binary']")

def compute_metrics(preds, labels, task_type: str, metrics_type: str):
    """
    Compute metric on full prediction and label arrays.
    preds: numpy array or torch tensor of shape (n_samples, ...) or (n_samples,) for classification/regression
    labels: same shape as preds for labels
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else labels
        
    if task_type == 'classification':
        if metrics_type == 'accuracy':
            # preds: logits or probabilities, labels: integers
            y_pred = preds.argmax(-1)
            correct = (y_pred == y_true).sum()
            return correct / len(y_true)
    
    elif task_type == 'binary':
        # preds: real values or logits, labels: {0,1}
        mask = y_true != -1
        if metrics_type == 'mcc':
            y_pred = (preds > 0).astype(int)
            return metrics.matthews_corrcoef(y_true[mask], y_pred[mask])
        elif metrics_type == 'auroc':
            return metrics.roc_auc_score(y_true[mask], preds[mask])
        elif metrics_type == 'auprc':
            return metrics.average_precision_score(y_true[mask], preds[mask])
        elif metrics_type == 'accuracy':
            y_pred = (preds > 0).astype('float32')
            return metrics.accuracy_score(y_true[mask], y_pred[mask])
    
    elif task_type == 'multilabel':
        if metrics_type == 'fmax':
            # preds: logits, labels: multi-hot
            y_pred = expit(preds) # sigmoid function
            return fmax(y_pred, y_true)
    
    elif task_type == 'regression':
        # print(np.mean(y_true), np.mean(preds))
        # print(np.std(y_true), np.std(preds))
        if metrics_type == 'spearmanr':
            return spearmanr(y_true, preds).correlation
        elif metrics_type == 'neg_mse':
            return -metrics.mean_squared_error(y_true, preds)
        elif metrics_type == 'neg_mae':
            return -metrics.mean_absolute_error(y_true, preds)
        elif metrics_type == 'r2':
            return metrics.r2_score(y_true, preds)
        else:
            raise ValueError(f"Unknown metric: {metrics_type}, must be one of ['spearmanr', 'mse', 'mae', 'r2']")

    else:
        raise ValueError(f"Unknown task_type: {task_type}, task_type must be one of ['classification', 'regression', 'multilabel', 'binary']")

