import numpy as np
import keras
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import r2_score
predictions = "/home/dev/research/distopia/experiments-workbench/experiments/logs/distopia_1d_cnn_3_val_trimmed_regression_9/test_predictions.npy"
targets = "/home/dev/research/distopia/experiments-workbench/experiments/logs/distopia_1d_cnn_3_val_trimmed_regression_9/test_labels.npy"
inputs = "/home/dev/research/distopia/experiments-workbench/experiments/logs/distopia_1d_cnn_3_val_trimmed_regression_9/test_inputs.npy"


def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

# def r_square_np(y_true, y_pred):
#     res = np.sum(np.square(y_true - y_pred))
#     import pdb; pdb.set_trace()
#     tot = np.sum(np.square(y_true -np.mean(y_true)))
#     return 1 - (res/tot)

def r_square_np(y_true, y_pred):
    res = np.sum(np.square(y_true - y_pred),axis=0)
    import pdb; pdb.set_trace()
    tot = np.sum(np.square(y_true -np.mean(y_true,axis=0)),axis=0)
    return 1 - (res/tot)

pred = np.load(predictions)
targ = np.load(targets)
x = np.load(inputs)

print(r_square_np(targ,pred))
print("R^2 score: {}".format(r2_score(targ,pred)))
import pdb; pdb.set_trace()

outcomes = ['population','wasted votes','compactness']
outcomes = range(3)

tasks = [[0,0,1],[0,-1,1],[0,-1,0],[-1,0,1],[-1,0,0],[-1,-1,1],[-1,-1,0]]
raw = []
means = []
x_means = []
for task in tasks:
    idx = np.where([np.all(i) for i in targ == task])[0]
    preds = pred[idx]
    print(idx[-1])
    print("{} : {}".format(task,np.mean(preds,0)))
    means.append(np.mean(preds,0))
    x_vals = x[idx]
    raw.append(preds)
    x_means.append(np.mean(x_vals.reshape(x_vals.shape[0]*x_vals.shape[1],3),0))
    targets = targ[idx]
    print("{}:{}".format(task,r2_score(targets,preds)))
    #import pdb; pdb.set_trace()

fig,axs = plt.subplots(len(tasks),3, sharey=True, constrained_layout=True)
for i,ax_row in enumerate(axs):
    ax_row[0].bar(outcomes,tasks[i])
    ax_row[1].bar(outcomes,means[i],color='g')
    import pdb; pdb.set_trace()
    ax_row[2].boxplot(raw[i])
    #ax_row[2].bar([0,1,2],[-4.0/7,-4.0/7,4.0/7])
    cs1 = "Cosine Similarity: {0:.2f}".format(cosine_similarity(np.array(means[i]).reshape(1,-1),np.array(tasks[i]).reshape(1,-1))[0][0])
    #cs2 = "Cosine Similarity: {0:.2f}".format(cosine_similarity(np.array(x_means[i]).reshape(1,-1),np.array(tasks[i]).reshape(1,-1))[0][0])
    ax_row[1].text(1.5,-0.8,cs1)
    #ax_row[2].bar(outcomes,x_means[i])
    #ax_row[2].text(1.5,-0.8,cs2)
    for ax in ax_row:
        ax.axhline(0)
        ax.set_title(tasks[i])
        ax.set_xticks([])
#plt.tight_layout()
plt.show()
import pdb; pdb.set_trace()     