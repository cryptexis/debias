import itertools
import matplotlib.pyplot as plt
import numpy as np
from utils.data import breakdown_df_shapes


def plot_confusion_matrix(cm, title='Confusion matrix'):
    """
    Plot the confusion matrix from aggregated data
    :param cm: aggregated matrix in the form of [[tp, fp], [fn, tn]]
    :param title: Title of the plot
    :return:
    """
    # set the coloring scheme
    cmap = plt.get_cmap('Blues')

    # configure the plot size
    plt.figure(figsize=(12, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18)
    plt.colorbar()

    # iterate over rows and columns of the cm
    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.1f}".format(cm[i, j]),
                 horizontalalignment="center",
                 fontsize=16,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Predicted label', fontsize=16)
    plt.xlabel('True label', fontsize=16)
    plt.yticks([])
    plt.xticks([])
    plt.show()


def plot_bar(keys, values, title, threshold = 0.1):
    """
    Bar plot from aggregated data
    :param keys: list of labels to be applied
    :param values: list of values to plot the bar
    :param title: The  title of the plot
    :param threshold: limiting value to label with "bias"
    :return:
    """
    # configure the plot
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_ylim([-1, 1])

    # draw horizontal lines
    for i in [-1.0, -0.5, 0.0, 0.5, 1]:
        plt.axhline(y=i, color='grey', linestyle='--')
    ax.bar(keys, values)
    ax.set_title(title)

    rectangles = ax.patches

    # label bar with red "bias" term when threshold is exceeded
    for rect in rectangles:
        height = rect.get_height()
        if np.abs(height) > threshold:
            ax.text(rect.get_x() + rect.get_width() / 2, height, "Bias",
                    ha='center', va='bottom', color='r')
    plt.show()


def plot_breakdown_confusion_matrix(global_tp, global_tn, global_fp, global_fn, col, val):
    """
    Plots the confusion matrix computed on a subset
    :param global_tp: Dataframe with all true_positives
    :param global_tn: Dataframe with all true_negatives
    :param global_fp: Dataframe with all false_positives
    :param global_fn: Dataframe with all false_negatives
    :param col: Candidate column
    :param val: Value of the column
    :return:
    """
    tp_local, tn_local, fp_local, fn_local = breakdown_df_shapes(global_tp, global_tn, global_fp, global_fn, col, val)

    cm = np.array([[tp_local, fp_local],
                   [fn_local, tn_local]])

    plot_confusion_matrix(cm=cm, title=f"Confusion Matrix {val}")
