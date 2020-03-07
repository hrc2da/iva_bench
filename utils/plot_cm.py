import os
import sys
import numpy as np
sys.path.append(os.path.normpath(os.getcwd() + os.sep + os.pardir))
from utils import plot_confusion_matrix

path = sys.argv[1]

targets = np.load(os.path.join(path,"test_labels.npy"))
predictions = np.load(os.path.join(path,"test_prediction_classes.npy"))
labels = np.load(os.path.join(path,"task_labels.npy"))
plot_confusion_matrix(targets,predictions,labels=labels)