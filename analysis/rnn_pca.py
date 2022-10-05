import numpy as np
import h5py
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA

file_name = './bptt_outputs.npz'

data_dict = np.load(file_name)

train_moutput=data_dict['train_output']
train_states=data_dict['train_states']
valid_moutput=data_dict['valid_output']
valid_states=data_dict['valid_states']

import pdb; pdb.set_trace()

with h5py.File('sim_task.h5', 'r') as h5f:
    train_input = h5f['train_input'][()]
    train_output = h5f['train_output'][()]
    valid_input = h5f['valid_input'][()]
    valid_output = h5f['valid_output'][()]
    train_inds = h5f['train_inds'][()]
    valid_inds = h5f['valid_inds'][()]

num_trials = train_input.shape[0] + valid_input.shape[0]

all_input = np.fill((num_trials, train_input.shape[1], train_input.shape[2]), np.nan)
all_input[train_inds] = train_input
all_input[valid_inds] = valid_input

all_output = np.fill((num_trials, train_output.shape[1], train_output.shape[2]), np.nan)
all_output[train_inds] = train_output
all_output[valid_inds] = valid_output

left_trials = all_output[:, -1, 0] < 0.
right_trials = all_output[:, -1, 0] > 0.

for i in range(0, train_moutput.shape[0], 50):
    exit(0)
