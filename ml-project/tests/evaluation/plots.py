import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    save_path,
):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    ticks = range(len(class_names))
    plt.xticks(ticks, class_names, rotation=45)
    plt.yticks(ticks, class_names)

    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curves(
    y_true,
    y_prob,
    class_names,
    save_path,
):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    num_classes = len(class_names)
    y_true_onehot = np.eye(num_classes)[y_true]

    plt.figure(figsize=(8, 6))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(
            y_true_onehot[:, i],
            y_prob[:, i],
        )
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            label=f"{class_names[i]} (AUC={roc_auc:.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
