import numpy as np

predictions = np.load("logs/distopia_1d_cnn_3_val_trimmed_32/test_predictions.npy")
labels = np.load("logs/distopia_1d_cnn_3_val_trimmed_32/test_labels.npy")

firsts = 0
seconds = 0
thirds = 0
print(predictions[-1])
print(np.argmax(predictions[-1]))
for i,p in enumerate(predictions):
    label = labels[i]
    # get a list of the indices in min to max order
    # so index from the back, please
    ascending_sorted_indices = np.argsort(p)
    if ascending_sorted_indices[-1] == label:
        firsts += 1
        seconds += 1
        thirds += 1
    # be careful here: this should not run if it was the first choice
    # because if it's the first choice, it's faster not to check 2 and 3
    # since 1,2,3 are all true
    elif ascending_sorted_indices[-2] == label:
        seconds += 1
        thirds += 1
    elif ascending_sorted_indices[-3] == label:
        thirds += 1
    # else do nothing

print(firsts/len(labels))
print(seconds/len(labels))
print(thirds/len(labels))
    