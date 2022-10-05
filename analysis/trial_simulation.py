import numpy as np
import h5py

dt = 1. / 30.

num_trials = 200
stim_start = np.random.uniform(0.2, 0.5, size=(num_trials,))
stim_end = stim_start + 1.0
go_delay = 0.5
trial_end = 1.5 + go_delay + 1.5
stim_start_idx = np.round(stim_start / dt).astype(int)
stim_end_idx = np.round(stim_end / dt).astype(int)
go_delay_idx = np.round(go_delay / dt).astype(int)
tlen = int(round(trial_end / dt))

noise_std = np.array([
    0.0, # stim
    0.05, # fixation
    # 0.05, # go tone
])

stim_rates = [  6,   7,    8,    9,   10,   11,   12,   20,   21,   22,   23,   24,  25,  26]
stim_prop =  [0.2, 0.2, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.2, 0.2]
trial_stim = np.random.choice(a=stim_rates, size=num_trials, p=stim_prop)
trial_correct = np.where(trial_stim > 16, np.ones(num_trials), -1. * np.ones(num_trials))

stim_arr = np.ones((num_trials, tlen, 1)) * trial_stim[:, None, None] * dt
# stim_arr = np.ones((num_trials, tlen, 1)) * trial_stim[:, None, None] * 0.04
# stim_arr[:, :stim_start_idx, :] = 0.
# stim_arr[:, stim_end_idx:, :] = 0.
for i, (ssi, sei) in enumerate(zip(stim_start_idx, stim_end_idx)):
    stim_arr[i, :ssi, :] = 0.
    stim_arr[i, sei:, :] = 0.
stim_arr = np.random.poisson(stim_arr)

# start_signal = np.zeros((600, tlen, 1))
# start_signal[:, stim_start_idx - 1, :] = 1.

# go_signal = np.zeros((600, tlen, 1))
# go_signal[:, stim_end_idx + go_delay_idx, :] = 1.

# input_arr = np.dstack([stim_arr, start_signal, go_signal])
# input_noise = np.random.normal(loc=0., scale=1., size=input_arr.size) * noise_std[None, None, :]

fixation_signal = np.zeros((num_trials, tlen, 1))
# fixation_signal[:, :stim_end_idx + go_delay_idx, :] = 1.
for i, (ssi, sei) in enumerate(zip(stim_start_idx, stim_end_idx)):
    fixation_signal[i, (ssi-3):sei + go_delay_idx, :] = 1.

input_arr = np.dstack([stim_arr, fixation_signal])
input_noise = np.random.normal(loc=0., scale=1., size=input_arr.shape) * noise_std[None, None, :]

# output_arr = np.ones((num_trials, tlen, 1)) * trial_correct[:, None, None]
# output_arr[:, :stim_end_idx + go_delay_idx + 1, :] = 0.

output_signal = np.ones((num_trials, tlen, 1)) * trial_correct[:, None, None]
# output_signal[:, :stim_end_idx + go_delay_idx + 1, :] = 0.
for i, (ssi, sei) in enumerate(zip(stim_start_idx, stim_end_idx)):
    output_signal[i, :sei + go_delay_idx, :] = 0.

output_arr = np.dstack([output_signal, fixation_signal])

valid_ratio = 0.2
valid_inds = np.arange(0, num_trials, int(round(1 / valid_ratio)))
train_inds = np.array([n for n in range(num_trials) if n not in valid_inds])

print(input_arr[train_inds].shape)
print(output_arr[valid_inds].shape)

with h5py.File('sim_task.h5', 'w') as h5f:
    h5f.create_dataset('train_input', data=input_arr[train_inds])
    h5f.create_dataset('valid_input', data=input_arr[valid_inds])
    h5f.create_dataset('train_output', data=output_arr[train_inds])
    h5f.create_dataset('valid_output', data=output_arr[valid_inds])
    h5f.create_dataset('train_inds', data=train_inds)
    h5f.create_dataset('valid_inds', data=valid_inds)
