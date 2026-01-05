import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
)


def compute_classification_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "report": classification_report(y_true, y_pred),
    }


def compute_roc_auc(y_true, y_prob, num_classes):
    """
    y_prob: shape (N, C), softmax probabilities
    """

    y_true_onehot = np.eye(num_classes)[y_true]

    auc_macro = roc_auc_score(
        y_true_onehot,
        y_prob,
        average="macro",
        multi_class="ovr",
    )

    auc_per_class = roc_auc_score(
        y_true_onehot,
        y_prob,
        average=None,
        multi_class="ovr",
    )

    return {
        "auc_macro": auc_macro,
        "auc_per_class": auc_per_class.tolist(),
    }
