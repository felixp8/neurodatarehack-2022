# %%

import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# %%

subject = 'sub-mouse1-fni16'
psth_dict = np.load(f'./{subject}_psth.npz')

# %%

with open(f'./{subject}_dates.txt', 'r') as f:
    sess_lines = f.readlines()

sess_list = [line.partition(':')[0] for line in sess_lines]
sess_list = sess_list[:26]

# %%

low_psth_good = psth_dict[f'{sess_list[-1]}_low_psth']
high_psth_good = psth_dict[f'{sess_list[-1]}_high_psth']

stacked = np.vstack([low_psth_good, high_psth_good])
norm_stacked = StandardScaler().fit_transform(stacked)

# %%

traj = PCA(n_components=5).fit_transform(stacked)
norm_traj = PCA(n_components=5).fit_transform(norm_stacked)

# %%

low_traj, high_traj = np.split(traj, [low_psth_good.shape[0]])
low_norm_traj, high_norm_traj = np.split(norm_traj, [low_psth_good.shape[0]])

# %%

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(low_traj[:, 1], low_traj[:, 0], low_traj[:, 2], color='blue')
ax.plot(high_traj[:, 1], high_traj[:, 0], high_traj[:, 2], color='red')
ax.set_xlabel('PC2')
ax.set_ylabel('PC1')
ax.set_zlabel('PC3')
plt.show()
plt.clf()

# %%

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(low_norm_traj[:, 1], low_norm_traj[:, 0], low_norm_traj[:, 2], color='blue')
ax.plot(high_norm_traj[:, 1], high_norm_traj[:, 0], high_norm_traj[:, 2], color='red')
ax.set_xlabel('PC2')
ax.set_ylabel('PC1')
ax.set_zlabel('PC3')
plt.show()
plt.clf()

# %%

plt.plot(low_norm_traj[:, 0], low_norm_traj[:, 1], 'b')
plt.plot(high_norm_traj[:, 0], high_norm_traj[:, 1], 'r')
plt.show()
plt.clf()

# %%

print(len(low_norm_traj))
print(len(high_norm_traj))

# %%

diff = np.linalg.norm(low_norm_traj - high_norm_traj, axis=1)
plt.plot(diff)
plt.show()
plt.clf()

# %%

low_psth_bad = psth_dict[f'{sess_list[0]}_low_psth']
high_psth_bad = psth_dict[f'{sess_list[0]}_high_psth']

bstacked = np.vstack([low_psth_bad, high_psth_bad])
norm_bstacked = StandardScaler().fit_transform(bstacked)

# %%

btraj = PCA(n_components=5).fit_transform(bstacked)
norm_btraj = PCA(n_components=5).fit_transform(norm_bstacked)

# %%

low_btraj, high_btraj = np.split(btraj, [low_psth_bad.shape[0]])
low_norm_btraj, high_norm_btraj = np.split(norm_btraj, [low_psth_bad.shape[0]])

# %%

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(low_btraj[:, 1], low_btraj[:, 0], low_btraj[:, 2], color='blue')
ax.plot(high_btraj[:, 1], high_btraj[:, 0], high_btraj[:, 2], color='red')
ax.set_xlabel('PC2')
ax.set_ylabel('PC1')
ax.set_zlabel('PC3')
plt.show()
plt.clf()

# %%

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(low_norm_btraj[:, 1], low_norm_btraj[:, 0], low_norm_btraj[:, 2], color='blue')
ax.plot(high_norm_btraj[:, 1], high_norm_btraj[:, 0], high_norm_btraj[:, 2], color='red')
ax.set_xlabel('PC2')
ax.set_ylabel('PC1')
ax.set_zlabel('PC3')
plt.show()
plt.clf()

# %%

plt.plot(low_norm_btraj[:, 0], low_norm_btraj[:, 1], 'b')
plt.plot(high_norm_btraj[:, 0], high_norm_btraj[:, 1], 'r')
plt.show()
plt.clf()

# %%

print(len(low_norm_btraj))
print(len(high_norm_btraj))

# %%

diff = np.linalg.norm(low_norm_btraj - high_norm_btraj, axis=1)
plt.plot(diff)
plt.show()
plt.clf()

# %%
