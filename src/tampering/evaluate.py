import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate(model, X, y):
    y_preds = model.predict_proba(X)
    is_binary = y_preds.shape[1] == 2
    multiclass = None if is_binary else "ovr"
    average_types = ["binary"] if is_binary else ["micro", "macro", "weighted"]
    metrics = {"accuracy": compute_accuracy(y, y_preds)}
    precision = {
        f"precision_{key}": compute_precision_score(y, y_preds, average=key)
        for key in average_types
    }
    recall = {
        f"recall_{key}": compute_recall_score(y, y_preds, average=key)
        for key in average_types
    }
    f1_metrics = {
        f"f1_{key}": compute_f1_score(y, y_preds, average=key) for key in average_types
    }
    try:
        roc_auc_scores = (
            {
                f"roc_auc_{key}": compute_roc_auc(
                    y, y_preds, average=key, multi_class=multiclass
                )
                for key in ["macro", "weighted"]
            }
            if not is_binary
            else {"roc_auc": roc_auc_score(y, y_preds[:, 1])}
        )
    except:
        roc_auc_scores = {"roc_auc": 0}
    metrics.update(precision)
    metrics.update(recall)
    metrics.update(f1_metrics)
    metrics.update(roc_auc_scores)
    metrics["matrix"] = confusion_matrix(y, np.argmax(y_preds, axis=1))
    return metrics


def compute_precision_score(y_true, y_pred, average):
    y_pred_argmax = y_pred.argmax(axis=1)
    return precision_score(y_true, y_pred_argmax, average=average)


def compute_recall_score(y_true, y_pred, average):
    y_pred_argmax = y_pred.argmax(axis=1)
    return recall_score(y_true, y_pred_argmax, average=average)


def compute_f1_score(y_true, y_pred, average):
    y_pred_argmax = y_pred.argmax(axis=1)
    return f1_score(y_true, y_pred_argmax, average=average)


def compute_roc_auc(y_true, y_pred, average, multi_class):
    return roc_auc_score(y_true, y_pred, average=average, multi_class=multi_class)


def compute_accuracy(y_true, y_pred):
    y_pred_argmax = y_pred.argmax(axis=1)
    y_comp = y_pred_argmax == y_true
    sum_of_correct_predictions = np.sum(y_comp)
    acc = sum_of_correct_predictions / y_true.shape[0]
    return acc


def compute_roc_curve(y_true, y_predict):
    return roc_curve(y_true, y_predict)


def compute_auc(fpr, tpr):
    return auc(fpr, tpr)


def compute_binary_roc_curve(y_true, y_predict):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_predict.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr, tpr, roc_auc


def compute_multiclass_roc_curve(y_true, y_predict, n_classes):
    # See https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#plot-roc-curves-for-the-multiclass-problem
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = compute_roc_curve(y_true[:, i], y_predict[:, i])
        roc_auc[i] = compute_auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_predict.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = compute_auc(fpr["macro"], tpr["macro"])
    return fpr, tpr, roc_auc


def compute_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, np.argmax(y_pred, axis=1))
    return matrix
