from matplotlib import pyplot as plt

import numpy as np

x_train = np.load("/home/dev/research/distopia/experiments-workbench/viz/x_train_flat.npy")
y_train = np.load("/home/dev/research/distopia/experiments-workbench/viz/y_train_flat.npy")

x_hum = np.load("/home/dev/research/distopia/experiments-workbench/viz/x_test_flat.npy")
y_hum = np.load("/home/dev/research/distopia/experiments-workbench/viz/y_test_flat.npy")

tasks = [[0,0,1],[0,-1,1],[0,-1,0],[-1,0,1],[-1,0,0],[-1,-1,1],[-1,-1,0]]
mean_windows = []
hum_windows = []
for task in tasks:
    idx = np.where([np.all(i) for i in y_train == task])[0]
    x_task = x_train[idx]
    x_task = x_task.reshape(130,100,3)
    mean_windows.append(np.mean(x_task,0))

    hdx = np.where([np.all(i) for i in y_hum == task])[0]
    hum_task = x_hum[hdx]
    hum_task = hum_task.reshape(10,100,3)
    hum_windows.append(np.mean(hum_task,0))

fig,axs = plt.subplots(len(tasks),2,sharey=True, constrained_layout=True)


for i,ax in enumerate(axs):
    ax1,ax2 = ax
    mw = mean_windows[i]
    ax1.plot(mw[:,0],label='population')
    ax1.plot(mw[:,1],label='wasted votes')
    ax1.plot(mw[:,2],label='compactness')
    ax1.set_title(tasks[i])
    ax1.set_ylim([-2,2])

    hw = hum_windows[i]
    ax2.plot(hw[:,0],label='population')
    ax2.plot(hw[:,1],label='wasted votes')
    ax2.plot(hw[:,2],label='compactness')
    ax2.set_title(tasks[i])
    ax2.set_ylim([-2,2])

plt.show()
