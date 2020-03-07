from keras import Model
from keras.models import load_model
import numpy as np
import yaml
import sys
import os
sys.path.append(os.path.normpath(os.getcwd() + os.sep + os.pardir))
from data_types.distopia_data import DistopiaData

'''
Take a model with a softmax and peel it back so you get the hidden layer as an output
'''
experiment_dir = "/home/dev/research/distopia/experiments-workbench/experiments/logs/distopia_1d_cnn_3_val_trimmed_regression_3/"
saved_model_path = "/home/dev/research/distopia/experiments-workbench/experiments/logs/distopia_1d_cnn_3_val_trimmed_regression_3/model.h5"
test_data_path = "/home/dev/research/distopia/experiments-workbench/experiments/logs/distopia_1d_cnn_3_val_trimmed_regression_3/test_inputs.npy"#/home/dev/data/distopia/team_logs/team_merged_introspective_standardized.npy"
test_labels_path = "/home/dev/research/distopia/experiments-workbench/experiments/logs/distopia_1d_cnn_3_val_trimmed_regression_3/test_labels.npy"#"/home/dev/data/distopia/team_logs/team_merged_labels.npy"


base_model = load_model(saved_model_path)

test_data = np.load(test_data_path)
test_labels = np.load(test_labels_path)

import pdb; pdb.set_trace()

#new_last_layer = base_model.layers[-(layers_to_strip+1)].name

#peeled_model = Model(inputs = base_model.input, outputs = base_model.get_layer(new_last_layer).output)

# test_data = np.load(test_data_path)
# test_data = test_data.reshape(1,test_data.shape[0],test_data.shape[1])
# test_labels = np.load(test_labels_path)

with open(experiment_dir+"distopia_1d_cnn_3_val_trimmed_32.spec.yaml", 'r') as stream:
    try:
        specs = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
data_specs = specs["data"]
test_data = DistopiaData()
test_data.set_params(data_specs)
test_data.preprocessors = ['filter_by_task','filter_by_metrics','sliding_window','onehot2class','balance_samples','class2onehot'] #data_specs["test_preprocessors"] #hackity hack hack hack
if "test_labels_path" in data_specs:
    test_labels_path = data_specs["test_labels_path"]
else:
    test_labels_path = None
test_data.load_data(test_data_path, labels_path=test_labels_path)
x_test = test_data.x
y_test = test_data.y
 
peeled_model.compile(optimizer="adam",loss="mse",metrics=["accuracy","cosine_proximity"])
peeled_model.evaluate(x_test,y_test, verbose=1)
import pdb; pdb.set_trace()