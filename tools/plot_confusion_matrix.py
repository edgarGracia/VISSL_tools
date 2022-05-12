import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import itertools
import seaborn as sn
import pandas as pd


def plot_confusion_matrix(cm: np.ndarray, target_names: list, output_path: Path,
    title: str ='', normalize: bool = True):
    """ Makes a plot from a sklearn confusion matrix.

    Args:
        cm (np.ndarray): The sklearn confusion matrix.
        target_names (list): List of targets name. e.g. ['cat', 'dog']
        output_path (Path): Output image path.
        title (str, optional): Title of the plot.
        normalize (bool, optional): Normalize the matrix values. Defaults True.
    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
	
    cm_orig = cm.copy()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    s = max(10, len(target_names) * 2)
    plt.figure(figsize=(s, s))
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    if title:
        plt.title(title)
    plt.colorbar(im,fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45, ha='right')
    plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, f"{round(cm[i, j], 2)} ({cm_orig[i, j]})",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, str(cm[i, j]), horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    fig = plt.gcf()

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    
    fig.savefig(output_path, dpi=100)
