import numpy as np
import os
import glob

base = r'y:\BIC\DHP19EPC_dataset\custom\TRANSPORTING_TR_2'
cam2_files = sorted(glob.glob(os.path.join(base, '*cam2.npy')))
print(f'Cam2 Count: {len(cam2_files)}')

for idx, f in enumerate(cam2_files):
    fname = os.path.basename(f)
    # fname: frame_000000_cam2.npy
    parts = fname.split('_')
    fid = parts[1] # 000000
    
    lab_f = os.path.join(base, f'frame_{fid}_label.npy')
    
    if os.path.exists(lab_f):
        d = np.load(lab_f)
        max_conf = np.max(d[2, :])
        if max_conf > 0.5:
            print(f'FOUND VALID PAIR at Index {idx}, Frame {fid}, MaxConf: {max_conf}')
            break
