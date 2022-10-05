import pandas as pd
import numpy as np
import os
import traceback

df_dir = './roi_info/sub-mouse1-fni16/'

roi_dfs = []

for fname in os.listdir(df_dir):
    roi_df = pd.read_csv(os.path.join(df_dir, fname))
    roi_dfs.append(roi_df)

shared_rois = np.arange(roi_dfs[0].roi_id.max() + 100)

for roi_df in roi_dfs:
    mask = np.isin(shared_rois, roi_df[roi_df.roi_status == 'good'].roi_id)
    shared_rois = shared_rois[mask]

print(len(shared_rois))

for rid in range(700):
    neur_types = []
    for roi_df in roi_dfs:
        try:
            neur_type = roi_df[roi_df.roi_id == rid].neuron_type.to_numpy()[0]
        except:
            traceback.print_exc()
            import pdb; pdb.set_trace()
        neur_types.append(neur_type)
    if len(np.unique(neur_types)) > 1:
        if len(np.unique(neur_types)) == 2:
            print(rid, end=', ')
        else:
            import pdb; pdb.set_trace()