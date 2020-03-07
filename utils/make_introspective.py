# input_path = "/home/dev/data/distopia/greedy_logs/3_obj/agent_merged.npy"
# output_path = "/home/dev/data/distopia/greedy_logs/3_obj/agent_merged_introspective_standardized.npy"

#hijacking this to make a human standardization params file
input_path = "/home/dev/data/distopia/team_logs/team_merged.npy"
output_path = "/home/dev/data/distopia/team_logs/team_standardization_params.pkl"


import numpy as np
import pickle as pkl

data = np.load(input_path)
data_m = np.mean(data,0)
data_std = np.std(data,0)
data_standard = (data-data_m)/data_std

#np.save(output_path,data_standard)

with open(output_path,'wb') as outfile:
    pkl.dump((data_m,data_std),outfile)