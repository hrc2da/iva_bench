from matplotlib import pyplot as plt
import pickle as pkl


history_f = "/home/dev/research/distopia/experiments-workbench/experiments/logs/distopia_1d_cnn_3_val_trimmed_32_repeat/history.pkl"
# history_f = "/home/dev/research/distopia/experiments-workbench/experiments/logs/distopia_1d_cnn_3_val_trimmed_regression_9/history.pkl"
with open(history_f,'rb') as infile:
    history = pkl.load(infile)


#import pdb; pdb.set_trace()
# plt.plot(history['accuracy'], label='accuracy')
# plt.plot(history['val_accuracy'], label='validation accuracy')
plt.plot(history['loss'], label='cross-entropy loss')
plt.plot(history['val_loss'], label='validation cross-entropy loss')
# plt.plot(history['mean_squared_error'], label='mean squared error')
# plt.plot(history['val_mean_squared_error'], label='validation mean squared error')
# plt.plot(history['cosine_proximity'], label='cosine proximity')
# plt.plot(history['val_cosine_proximity'], label='validation cosine proximity')
#plt.ylim([0,4])
#plt.xlim([0,100])
plt.legend(loc=0)
plt.xlabel("Epoch")
plt.xlim([-1,100])
plt.ylim([0,3])
plt.show()

import pdb; pdb.set_trace()

