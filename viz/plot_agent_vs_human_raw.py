#this does not slide windows. it just takes as much human data as possible and averages it out over that time


# process the human files
'''
Strategy
1. Go through each file and standardize (for the plot)
2. update a count and a sum for each index (step0, step1, etc) for each task (use a dict)
3. divide the sum by the count for each index
4. plot
'''

from pathlib import Path
import sys
import os
sys.path.append(os.path.normpath(os.getcwd() + os.sep + os.pardir))
from data_types.distopia_data import DistopiaData
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt


# generate a mean/std vector for all metrics from a set of agent json log files
# if (len(sys.argv) != 2):
#     print("USAGE: python distopia_generate_standardization_file.py <data path>")
#     exit(0)
data_dir = "/home/dev/data/distopia/team_logs/"
standardization_file = "/home/dev/data/distopia/more_random/logs/mr_standardization_params.pkl"
logpaths = sorted(Path(data_dir).glob('*.json'))
print("Found {} logfiles to process.".format(len(logpaths)))

# tasks = (['[ -1. -1.  0.]', '[-1.  -1.  1.]', '[-1. 0. 0.]', '[-1. 0. 1.]', '[0.  -1.  0.]', '[ 0. -1.  1.]', '[ 0.  0. 1.]'])
task_arr = [[0.,0.,1.],[0.,-1.,1.],[0.,-1.,0.],[-1.,0.,1.],[-1.,0.,0.],[-1.,-1.,1.],[-1.,-1.,0.]]
data = DistopiaData()
tasks = [data.task_arr2str(task) for task in task_arr]
cur_metrics = ['population', 'pvi', 'compactness']
data.set_params({'metric_names': cur_metrics,'standardization_file':standardization_file,'preprocessors':['standardize']})
# let's be expensive and run through the files twice; the first time will be to get the longest sequence so I can pre-allocate the array
# the second time will be to update it.
# nevermind, let's take the first 100, since this matches the agent
max_len = 100
lengths = []
sequence_dict = {task: {'cumsum': np.zeros((max_len,3)), 'counts': np.zeros((max_len,3)), 'max_index':0} for task in tasks}

for log in logpaths:
    # reset the agent between loads, so I can update my running average array
    data.load_data(log,append=False,load_designs=False,load_metrics=True)
    task_dict = data.get_task_dict()
    for task,sequences in task_dict.items():
        if task not in tasks:
            continue

        for sequence in sequences:
            cutoff = min(max_len, len(sequence))
            if cutoff > sequence_dict[task]['max_index']:
                sequence_dict[task]['max_index'] = cutoff # I actually don't need this
            subsequence = sequence[:cutoff]
            # update sequence counts
            for i in range(cutoff):
                sequence_dict[task]['counts'][i] += [1,1,1]
                sequence_dict[task]['cumsum'][i] += subsequence[i]

avg_sequence_dict = {task:np.zeros((max_len,3)) for task in tasks}

for task,stats in sequence_dict.items():
    avg_sequence_dict[task] = stats['cumsum']/stats['counts']
    assert avg_sequence_dict[task].shape == (max_len,3)
    #data.load_data(log,append=True,load_designs=False,load_metrics=True)




hum_windows = []
for task in tasks:
    hum_windows.append(avg_sequence_dict[task])
hum_windows = np.array(hum_windows)

# now get the agent data
x_train = np.load("/home/dev/research/distopia/experiments-workbench/viz/x_train_flat.npy")
y_train = np.load("/home/dev/research/distopia/experiments-workbench/viz/y_train_flat.npy")
tasks = [[0,0,1],[0,-1,1],[0,-1,0],[-1,0,1],[-1,0,0],[-1,-1,1],[-1,-1,0]]

mean_windows = []

for task in tasks:
    idx = np.where([np.all(i) for i in y_train == task])[0]
    x_task = x_train[idx]
    x_task = x_task.reshape(130,100,3)
    mean_windows.append(np.mean(x_task,0))

np.save("./mean_100.npy",mean_windows)
np.save("./hum_100.npy",hum_windows)
import pdb; pdb.set_trace()
fig,axs = plt.subplots(len(tasks),2,sharey=True, constrained_layout=True)
for i,ax in enumerate(axs):
    ax1,ax2 = ax
    mw = mean_windows[i]
    ax1.plot(mw[:,0],label='population')
    ax1.plot(mw[:,1],label='wasted votes')
    ax1.plot(mw[:,2],label='compactness')
    ax1.set_title(tasks[i])
    ax1.set_ylim([-2,2])
    ax1.set_xlim([0,95])

    hw = hum_windows[i]
    ax2.plot(hw[:,0],label='population')
    ax2.plot(hw[:,1],label='wasted votes')
    ax2.plot(hw[:,2],label='compactness')
    ax2.set_title(tasks[i])
    ax2.set_ylim([-2,2])
    ax2.set_xlim([0,95])

plt.show()


