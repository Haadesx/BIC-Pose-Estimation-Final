
import numpy as np
import glob
import os
import argparse
from tqdm import tqdm

def check_labels(root_dir):
    # Pattern: *_label.npy in root_dir
    files = glob.glob(os.path.join(root_dir, "*_label.npy"))
    print(f"Found {len(files)} label files.")
    
    valid_count = 0
    zero_count = 0
    
    # Check a sample if too many
    if len(files) > 1000:
        files = files[:1000]
        print("Checking first 1000 files...")
        
    for f in tqdm(files):
        try:
            data = np.load(f) # [3, 13]
            # u, v, mask
            u = data[0]
            v = data[1]
            if np.sum(u) > 0 or np.sum(v) > 0:
                valid_count += 1
            else:
                zero_count += 1
        except:
            pass
            
    print(f"Summary: {valid_count} Valid (non-zero), {zero_count} Zero-filled.")

if __name__ == "__main__":
    check_labels(r'y:\BIC\DHP19EPC_dataset')
