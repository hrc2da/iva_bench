from importlib import import_module
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import os

def split_package_module(full_path):
    components = full_path.split('.')
    package = components[0]
    module = '.'.join(components[1:])
    return package,module

def split_module_package(full_path):
    components = full_path.split('.')
    package = components[0]
    module = '.'.join(components[1:])
    return module,package

def split_module_class(full_path):
    components = full_path.split('.')
    class_name = components[-1]
    module_name = '.'.join(components[:-1])
    return class_name,module_name

def import_class(full_path):
    class_name,module_name = split_module_class(full_path)
    return getattr(import_module(module_name), class_name)

def hierarchical_sort(lst, direction='lr', order='descending'):
    assert len(lst) > 0
    to_sort = lst[:]
    sample = to_sort[0]
    sort_order = -1 if order == 'descending' else 1
    if direction == 'rl':
        for i in range(len(sample)):
            to_sort.sort(key=lambda x: sort_order * x[i])
    elif direction == 'lr':
        for j in range(len(sample)-1, -1, -1):
            to_sort.sort(key=lambda x: sort_order * x[j])
    return to_sort


def plot_confusion_matrix(targets, predictions, labels=None, path=None):

    cm = confusion_matrix(targets, predictions)

    fig = plt.figure(figsize=(7, 6))
    cm = cm.astype(float)
    # row-normalize
    for row in range(cm.shape[0]):
        tot_freq = np.sum(cm[row, :])
        cm[row,:] = cm[row,:]/tot_freq
    if labels is not None:
        sns.heatmap(cm, cmap='YlGnBu', square=1, annot=True, linewidths=1, xticklabels=labels, yticklabels=labels)
    else:
        sns.heatmap(cm, cmap='YlGnBu', square=1)
    #sns.heatmap(cm[ymin:ymax, xmin:xmax], cmap='YlGnBu', square=1, annot=False, xticklabels=task_names[xmin:xmax], yticklabels=task_names[ymin:ymax])
    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(width=0)
    ax.xaxis.set_label_position('top')
    # plt.xticks(np.arange(len(task_names)), task_names, rotation=90)
    #plt.xlim(0, len(task_names))
    #plt.yticks(np.arange(len(task_names)), task_names, rotate=90)
    plt.ylim(cm.shape[0]+0.5, 0)  # inverted range for "upper" origin
    plt.ylabel('Target')
    plt.xlabel('Prediction')
    if path is None:
        plt.show()
    else:
        plt.savefig(os.path.join(path,'confusion_matrix.png'))
