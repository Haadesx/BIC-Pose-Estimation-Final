
import sys
import os
import glob
import cv2
import argparse
import subprocess
import torch
import numpy as np
import copy
from tqdm import tqdm

# ... (rest of imports/helpers)

def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Load Model
    model = Pose_PointNet(args).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    # Load Data
    # Expecting a glob pattern or single file
    if '*' in args.input_data:
        files = sorted(glob.glob(args.input_data))
    else:
        files = [args.input_data]
    
    if len(files) == 0:
        print("No input files found.")
        return

    # Use DHP19EPC preprocessing logic
    # We can instantiate dataset just to access its methods or reuse code
    # Hack: create lightweight dataset instance
    dataset_helper = DHP19EPC(args, root_data_dir='.', Test3D=True) # Params don't matter much for helper access
    
    all_events = []
    all_preds_x = []
    all_preds_y = []
    
    print(f"Processing {len(files)} frames...")
    
    with torch.no_grad():
        for f in tqdm(files, desc="Inference"):
            # Load raw events
            raw_data = np.load(f) # [N, 4]
            # Assumed format: x, y, t, p
            
            # Preprocess
            # Input to preprocess: [N, 4]
            # Output: [num_sample, C] -> [2048, 5]
            data = dataset_helper.RasEventCloud_preprocess(raw_data)
            
            # Batchify
            data_tensor = torch.from_numpy(data).float().unsqueeze(0).to(device) # [1, 2048, 5]
            data_tensor = data_tensor.transpose(2, 1) # [1, 5, 2048]
            
            # Inference
            pred_x, pred_y = model(data_tensor) # [1, num_joints, W], [1, num_joints, H]
            
            # Decode heatmaps to coords
            # decode_sa_simdr expects numpy
            px, py = decode_sa_simdr(pred_x.cpu().numpy(), pred_y.cpu().numpy())
            
            all_events.append(raw_data)
            all_preds_x.append(px)
            all_preds_y.append(py)
            
    # Visualize
    visualize_pose(all_events, all_preds_x, all_preds_y, args.output_video)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, required=True, help='Path to .npy event files (glob supported)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model .pth')
    parser.add_argument('--output_video', type=str, default='output_pose.mp4')
    
    # Model Args (Must match training)
    parser.add_argument('--num_joints', type=int, default=6)
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--sensor_sizeW', type=int, default=346)
    parser.add_argument('--sensor_sizeH', type=int, default=260)
    parser.add_argument('--label', type=str, default='mean', help="Required for Dataset init compatibility")
    parser.add_argument('--sigma', type=float, default=8)
    
    args = parser.parse_args()
    
    infer(args)
