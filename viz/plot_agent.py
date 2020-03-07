from matplotlib import pyplot as plt

import numpy as np

x_train = np.load("/home/dev/data/distopia/more_random/logs/one_hot_merged.npy")
y_train = np.load("/home/dev/data/distopia/more_random/logs/one_hot_merged_labels.npy")


#tasks = [[0,0,1],[0,-1,1],[0,-1,0],[-1,0,1],[-1,0,0],[-1,-1,1],[-1,-1,0]]
tasks = [[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]]
mean_windows = []
hum_windows = []
for task in tasks:
    idx = np.where([np.all(i) for i in y_train == task])[0]
    
    x_task = x_train[idx]
    print("{}:{}".format(task,x_task.shape))
    #x_task = x_task.reshape(130,100,3)
    #mean_windows.append(np.mean(x_task,0))
    mean_windows.append(x_task)



fig,axs = plt.subplots(len(tasks),sharey=True, constrained_layout=True)


for i,ax1 in enumerate(axs):
    mw = mean_windows[i]
    ax1.plot(mw[:,0],label='population')
    ax1.plot(mw[:,1],label='wasted votes')
    ax1.plot(mw[:,2],label='compactness')
    ax1.set_title(tasks[i])
    ax1.set_ylim([2,350000])

plt.legend()
plt.show()
