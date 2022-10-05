# %%
from pynwb import NWBHDF5IO
from dandi.dandiapi import DandiAPIClient
from sklearn.decomposition import PCA
from fsspec.implementations.cached import CachingFileSystem
import fsspec
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as signal
import os
import shutil
from tqdm import tqdm
from datetime import datetime
from dateutil.tz import tzoffset

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# %%

dandiset_id = '000016'
subject = 'sub-mouse2-fni17'
mouse_num = subject.split('-')[1]
print(mouse_num)

# mouse1-fni16
# first_day = datetime(2015, 8, 17, tzinfo=tzoffset(None, -14400))
# last_day = datetime(2015, 10, 29, tzinfo=tzoffset(None, -14400))
# exclude_days = [
#     datetime(2015, 9, 14, tzinfo=tzoffset(None, -14400)),
#     datetime(2015, 10, 15, tzinfo=tzoffset(None, -14400))
# ]

# mouse2-fni17
first_day = datetime(2015, 8, 14, tzinfo=tzoffset(None, -14400))
last_day = datetime(2015, 11, 2 + 1, tzinfo=tzoffset(None, -14400))
exclude_days = [
    datetime(2015, 9, 29, tzinfo=tzoffset(None, -14400)),
]

# mouse3-fni18
# first_day = datetime(2015, 8, 17, tzinfo=tzoffset(None, -14400))
# last_day = datetime(2015, 12, 17, tzinfo=tzoffset(None, -14400))
# exclude_days = [
# ]

fs = CachingFileSystem(fs=fsspec.filesystem("http"), cache_storage="nwb-cache",)

# %%

def get_timestamps(timeseries):
    if timeseries.timestamps is None:
        timestamps = np.arange(timeseries.data.shape[0]) * timeseries.rate + timeseries.starting_time
    else:
        timestamps = timeseries.timestamps
    return timestamps

# %%

def smooth_spikes(spikes, kern_sd, axis=0):
    kernel = signal.gaussian(int(6 * kern_sd), int(kern_sd), sym=True)
    filt = lambda x: np.convolve(x, kernel, 'same')
    smoothed = np.apply_along_axis(filt, axis, spikes)
    return smoothed

# %%

def get_trialsegmented_roi_timeseries(event_name, pre_stim_dur, post_stim_dur, smooth=True, trial_sel=None):
    event_roi_timeseries = nwbfile.modules['Ophys'].data_interfaces[event_name].roi_response_series
    tvec = get_timestamps(event_roi_timeseries.get('Trial_00'))  # timestamps from all trial are the same, so get one from trial_0
    # check if pre/post stim duration is out of bound
    pre_stim_dur = np.maximum(tvec[0], pre_stim_dur)
    post_stim_dur = np.minimum(tvec[-1], post_stim_dur)   
    # extract data
    ix = np.where(np.logical_and(tvec >= pre_stim_dur, tvec <= post_stim_dur))[0]
    # make trial mask
    if trial_sel is None:
        trial_sel = np.full((len(event_roi_timeseries)), True)
    else:
        assert len(trial_sel) == len(event_roi_timeseries)
    # stack data
    if smooth:
        return np.dstack([smooth_spikes(d.data[ix, :], 2) for i, d in enumerate(event_roi_timeseries.values()) if trial_sel[i]]), tvec[ix]   
    else:
        return np.dstack([d.data[ix, :] for i, d in enumerate(event_roi_timeseries.values()) if trial_sel[i]]), tvec[ix]   


# %%

with DandiAPIClient() as client:
    asset_list = list(client.get_dandiset(dandiset_id, 'draft').get_assets_with_path_prefix(subject))
    s3_urls = [asset.get_content_url(follow_redirects=1, strip_query=True) for asset in asset_list]
    fpaths = [asset.path for asset in asset_list]
    idx = np.argsort(fpaths)
    s3_urls = [s3_urls[i] for i in idx]

# %%

sess_list = []
date_list = []
sr_list = []
rt_list = []
low_psth_list = []
high_psth_list = []
roi_df_list = []

for s3_url in tqdm(s3_urls):
    # print(s3_url, end=': ')
    f = fs.open(s3_url, 'rb')
    file = h5py.File(f, 'r')
    with NWBHDF5IO(file=file, mode='r', path=file.filename, load_namespaces=True) as io:
        nwbfile = io.read()

        sess_name = nwbfile.session_id
        sess_date = nwbfile.session_start_time
        if (sess_date < first_day) or (sess_date > last_day) or (sess_date in exclude_days):
            print(f'Skipping session {sess_date}')
        # print(sess_name, end='\n')
        else:
            trial_info = nwbfile.trials.to_dataframe()
            success_rate = np.sum((trial_info.trial_response == 'correct') & (trial_info.trial_is_good)) / len(trial_info)
            sr_list.append(success_rate)
            response_time = np.nanmean(trial_info[trial_info.trial_is_good].first_commit - trial_info[trial_info.trial_is_good].go_tone)
            rt_list.append(response_time)

            roi_idx = nwbfile.modules['Ophys'].data_interfaces['dFoF_initToneAl'].roi_response_series.get('Trial_00').rois.data[()]
            roi_df = nwbfile.modules['Ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation'].to_dataframe()
            # import pdb; pdb.set_trace()

            low_mask = (trial_info.trial_type == 'Low-rate') & (trial_info.trial_is_good) #  & (trial_info.trial_response == 'correct')
            high_mask = (trial_info.trial_type == 'High-rate') & (trial_info.trial_is_good) #  & (trial_info.trial_response == 'correct')
            setting_names = ['low_stimAl', 'high_stimAl']
            segmentation_settings = [
                {'event_name':'dFoF_stimAl_noEarlyDec', 'pre_stim_dur': -0.100, 'post_stim_dur': 10.000, 'trial_sel': low_mask},
                {'event_name':'dFoF_stimAl_noEarlyDec', 'pre_stim_dur': -0.100, 'post_stim_dur': 10.000, 'trial_sel': high_mask},
                # {'event_name':'dFoF_firstSideTryAl', 'pre_stim_dur': -0.250, 'post_stim_dur': 0.250},
            ]

            trial_avg_segments = {}
            for name, setting in zip(setting_names, segmentation_settings):
                # extract segment
                out = get_trialsegmented_roi_timeseries(**setting)
                # average over trial
                trial_avg_segments[name] = (np.nanmean(out[0], axis=2), out[1])

            def roi_sort_by_peak_latency(roi_tcourse):
                sorted_roi_idx = np.argsort(np.argmax(roi_tcourse, axis = 1))
                return roi_tcourse[sorted_roi_idx,:].copy(), sorted_roi_idx

            # Concatenate and sort
            sorted_avg_segments = {k: (roi_sort_by_peak_latency(v[0].T)[0], v[1]) for k, v in trial_avg_segments.items()}

            # Concatenate all timevec(s) and determine the indices of t = 0
            # tvec_concat = [value[1] for value in trial_avg_segments.values()]
            # xdim_all = [t.size for t in tvec_concat]
            # xdim_all.insert(0,0)
            # zeros_all = [np.where(v == 0)[0] for v in tvec_concat]

            # Extract inh/exc status
            # is_inh = np.zeros((data_all.shape[0]))
            # is_inh[neuron_type[sorted_roi_idx] == 'inhibitory'] = 1

            fig1E = plt.figure(figsize=(16,12))

            for i in range(2):
                ax = fig1E.add_subplot(1, 2, i+1)
                ax.set_facecolor('white')

                data = sorted_avg_segments[sorted(list(sorted_avg_segments.keys()))[i]][0]
                sns.heatmap(data=data, xticklabels=[], yticklabels=[], cmap='YlGnBu_r', axes=ax, vmin=0)

                # add vertical lines
                # for zidx, z in enumerate(zeros_all):
                #     ax1.axvline(x=np.cumsum(xdim_all)[zidx], color='b',linestyle='-',linewidth=0.7)
                #     ax1.axvline(x=z + np.cumsum(xdim_all)[zidx], color='r',linestyle='--',linewidth=1)
            
                ax.set_xlim(0, data.shape[1]+10)
                ax.set_xlabel('Time')
                ax.set_ylabel('Neuron')
                ax.set_label('Averaged inferred spike for all neurons for an example session')

            plt.tight_layout()
            plt.savefig(f'./plots/psth_heatmaps_{mouse_num}/{sess_name}.png')
            plt.close()

            sess_list.append(sess_name)
            date_list.append(sess_date)
            low_psth_list.append(trial_avg_segments['low_stimAl'][0])
            high_psth_list.append(trial_avg_segments['high_stimAl'][0])
            roi_df_list.append(roi_df)

    file.close()
    f.close()
    
    for cache_obj in os.listdir('./nwb-cache/'):
        os.remove(f'./nwb-cache/{cache_obj}')

# %%

def reorder(l, idx):
    return [l[i] for i in idx]
sess_ordering = np.argsort(date_list)
sess_list = reorder(sess_list, sess_ordering)
date_list = reorder(date_list, sess_ordering)
sr_list = reorder(sr_list, sess_ordering)
rt_list = reorder(rt_list, sess_ordering)
low_psth_list = reorder(low_psth_list, sess_ordering)
high_psth_list = reorder(high_psth_list, sess_ordering)
roi_df_list = reorder(roi_df_list, sess_ordering)

# %%

with open(f'{subject}_dates.txt', 'w') as f:
    lines = [f'{name}: {date}\n' for name, date in zip(sess_list, date_list)]
    f.writelines(lines)

# %%

plt.plot(sr_list)
plt.xlabel('session')
plt.ylabel('success rate')
plt.ylim(0,1)
plt.savefig(f'./plots/succ_rate_{mouse_num}.png')
plt.clf()

# %%

plt.plot(rt_list)
plt.xlabel('session')
plt.ylabel('reaction time')
plt.savefig(f'./plots/reaction_time_{mouse_num}.png')
plt.clf()

# %%

# low_splits = np.cumsum([arr.shape[0] for arr in low_psth_list])[:-1]
# low_stacked = np.vstack(low_psth_list)
# high_splits = np.cumsum([arr.shape[0] for arr in high_psth_list])[:-1]
# high_stacked = np.vstack(high_psth_list)
# splits = [low_stacked.shape[0]]
# stacked = np.vstack([low_stacked, high_stacked])

# pca = PCA(n_components=2)
# traj = pca.fit_transform(stacked)

# low_traj, high_traj = np.split(traj, splits, axis=0)
# low_sess_traj = np.split(low_traj, low_splits)
# high_sess_traj = np.split(low_traj, high_splits)

# %%

low_sess_traj = []
high_sess_traj = []
for name, low, high in zip(sess_list, low_psth_list, high_psth_list):
    stacked = np.vstack([low, high])
    traj = PCA(n_components=2).fit_transform(stacked)
    low_traj, high_traj = np.split(traj, [low.shape[0]])
    plt.plot(low_traj[:, 0], low_traj[:, 1], color='blue')
    plt.plot(high_traj[:, 0], high_traj[:, 1], color='red')
    plt.savefig(f'./plots/pca_traj_{mouse_num}/{name}.png')
    plt.clf()
    low_sess_traj.append(low_traj)
    high_sess_traj.append(high_traj)

low_psth_dict = {sn + '_low_psth': arr for sn, arr in zip(sess_list, low_psth_list)}
high_psth_dict = {sn + '_high_psth': arr for sn, arr in zip(sess_list, high_psth_list)}
np.savez(f'{subject}_psth.npz', **low_psth_dict, **high_psth_dict)

# %%

fig = plt.figure(figsize=(8,8))
for i, sess in enumerate(sess_list):
    opacity = (i+1) / len(sess_list)

    plt.plot(low_sess_traj[i][:, 0], low_sess_traj[i][:, 1], color='blue', alpha=opacity)
    plt.plot(high_sess_traj[i][:, 0], high_sess_traj[i][:, 1], color='red', alpha=opacity)

plt.savefig(f'./plots/pca_evolution_{mouse_num}.png')

# %%

for name, roi_df in zip(sess_list, roi_df_list):
    roi_df.to_csv(f'./roi_info/{subject}/{name}.csv')
