
import numpy as np
import glob
import os
import argparse

def inspect(path_pattern):
    files = sorted(glob.glob(path_pattern))
    print(f"Found {len(files)} files matching {path_pattern}")
    
    if not files:
        return

    # Inspect first 5 and last 5
    to_inspect = files[:5] + files[-5:]
    
    for f in to_inspect:
        print(f"\n--- Inspecting {os.path.basename(f)} ---")
        try:
            data = np.load(f)
            print(f"Shape: {data.shape}")
            if data.size == 0:
                print("EMPTY FILE")
                continue
                
            # data: [N, 4] -> x, y, t, p
            x = data[:, 0]
            y = data[:, 1]
            t = data[:, 2]
            p = data[:, 3]
            
            print(f"X range: [{x.min()}, {x.max()}]")
            print(f"Y range: [{y.min()}, {y.max()}]")
            print(f"T range: [{t.min()}, {t.max()}]")
            print(f"P range: [{p.min()}, {p.max()}]")
            
            # Check against assumed DHP19 dims
            W, H = 346, 260
            invalid_x = np.sum(x >= W)
            invalid_y = np.sum(y >= H)
            print(f"Invalid X (>=346): {invalid_x} ({invalid_x/len(x)*100:.2f}%)")
            print(f"Invalid Y (>=260): {invalid_y} ({invalid_y/len(y)*100:.2f}%)")
            
            # Check for label
            # Assuming label is in parent dir? No, root_label_dir is passed in training
            # Label path logic from DHP19EPC.py:
            # label_name = data_name + '_label.npy'
            # label_file = os.path.join(self.root_label_dir, label_name)
            
            # We can try to guess label path
            # Parent of dvs-video is custom?
            # y:\BIC\DHP19EPC_dataset\custom\dvs-video -> y:\BIC\DHP19EPC_dataset
            
            # Logic: data_name is 'file_cam2'
            # label_name is 'file_cam2_label.npy'
            # path: y:\BIC\DHP19EPC_dataset\file_cam2_label.npy
            
            label_root = r'y:\BIC\DHP19EPC_dataset'
            label_name = os.path.basename(f).replace('.npy', '_label.npy')
            label_path = os.path.join(label_root, label_name)
            
            if os.path.exists(label_path):
                 l = np.load(label_path)
                 print(f"Label found: {l.shape}")
                 # l is [num_joints, 3] (u, v, mask)
                 u, v, mask = l[:].astype(float)
                 print(f"Label U range: [{u.min()}, {u.max()}]")
                 print(f"Label V range: [{v.min()}, {v.max()}]")
            else:
                 print(f"Label NOT found at {label_path}")

        except Exception as e:
            print(f"Error loading: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=r'y:\BIC\DHP19EPC_dataset\custom\dvs-video\*cam2.npy')
    args = parser.parse_args()
    inspect(args.path)
