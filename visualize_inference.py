import torch
import numpy as np
import cv2
import os
import argparse
import sys

sys.path.append(os.getcwd())

from dataset.DHP19EPC import DHP19EPC
from models.PointNet import Pose_PointNet

class Args:
    def __init__(self):
        self.num_points = 2048 # Match training
        self.label = 'last'
        self.sensor_sizeW = 346
        self.sensor_sizeH = 260
        self.sigma = 8
        self.num_joints = 6 

def visualize_inference():
    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = Pose_PointNet(args).to(device)
    checkpoint_path = r'y:\BIC\EventPointPose\checkpoints\pointnet_epoch_4.pth'
    if not os.path.exists(checkpoint_path):
        # Fallback to epoch 0 if 4 doesn't exist (though list_dir showed it)
        checkpoint_path = r'y:\BIC\EventPointPose\checkpoints\pointnet_epoch_0.pth'
    
    print(f"Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Load Dataset Item
    # Using TRANSPORTING_TR_2 frame 747 (dataset index) -> Frame 748 (filename)
    root_data = r"y:\BIC\DHP19EPC_dataset\custom\TRANSPORTING_TR_2"
    # Root label is same dir implies implicit
    
    # Manually instantiate DHP19EPC
    # Note: DHP19EPC.__init__ expects args, root_data_dir, root_label_dir
    # We need to hack the glob inside it or just use our updated class if we didn't update DHP19EPC.py itself consistently?
    # I updated DHP19EPC.py to use `os.path.dirname(file_dir)` for labels, so passing root_label_dir is less critical but required by Init signature.
    
    # BUT wait, DHP19EPC.py init does:
    # self.dvs_frames2 = sorted(glob.glob(os.path.join(root_data_dir, "*cam2.npy")))
    # It does NOT use recursive glob unless I passed the Custom logic.
    # The base class DHP19EPC.py I edited earlier:
    # I edited `load_sample` but did I edit `__init__` to recursive glob? 
    # Checking my history... I edited `train_arm_pose.py` to subclass and override `__init__`.
    # I did NOT edit `DHP19EPC.py`'s `__init__` to be recursive.
    # So if I use `DHP19EPC`, I must point it directly to the subdir.
    
    dataset = DHP19EPC(args, root_data_dir=root_data, root_label_dir=root_data, Test3D=False)
    
    # Index 747 corresponds to frame 748 in this specific folder (verified in prev steps)
    item_idx = 747
    
    print(f"Loading item {item_idx}...")
    data, x, y, weight = dataset[item_idx]
    
    # Prepare input
    # Data: [N, 5] -> [1, 5, N] ?
    # Train script: data = data.float().to(device).transpose(2, 1) -> [B, C, N]
    # DHP19EPC returns [N, 5]
    
    inputs = torch.from_numpy(data).float().unsqueeze(0).to(device) # [1, N, 5]
    inputs = inputs.transpose(2, 1) # [1, 5, N]
    
    # Inference
    with torch.no_grad():
        pred_x, pred_y = model(inputs)
        
    # Retrieve
    # Retrieve [num_joints, Size]
    pred_x_dist = pred_x.cpu().numpy()[0] 
    pred_y_dist = pred_y.cpu().numpy()[0]
    
    # SimDR Decoding: Argmax
    pred_x_coords = np.argmax(pred_x_dist, axis=-1)
    pred_y_coords = np.argmax(pred_y_dist, axis=-1)
    
    gt_x_dist = x
    gt_y_dist = y
    
    gt_x_coords = np.argmax(gt_x_dist, axis=-1)
    gt_y_coords = np.argmax(gt_y_dist, axis=-1)
    
    # Visualization
    sensor_w = args.sensor_sizeW
    sensor_h = args.sensor_sizeH
    canvas = np.zeros((sensor_h, sensor_w, 3), dtype=np.uint8)
    
    # Draw Events (Rasterize manually for viz as before)
    # data is [N, 5]. Col 0=x, 1=y.
    ev_x = data[:, 0].astype(int)
    ev_y = data[:, 1].astype(int)
    
    # Bounds
    mask = (ev_x >= 0) & (ev_x < sensor_w) & (ev_y >= 0) & (ev_y < sensor_h)
    ev_x = ev_x[mask]
    ev_y = ev_y[mask]
    
    canvas[ev_y, ev_x, 1] = 255 # Green
    
    # Draw GT (Yellow)
    for i in range(len(gt_x_coords)):
        gx, gy = int(gt_x_coords[i]), int(gt_y_coords[i])
        if gx > 0 and gy > 0:
            cv2.circle(canvas, (gx, gy), 4, (0, 255, 255), -1)
            
    # Draw Pred (Red)
    for i in range(len(pred_x_coords)):
        px, py = int(pred_x_coords[i]), int(pred_y_coords[i])
        # Preds might be out of bounds or crazy if untrained
        if 0 <= px < sensor_w and 0 <= py < sensor_h:
             cv2.circle(canvas, (px, py), 3, (0, 0, 255), -1)
             
    output_path = "inference_result.png"
    cv2.imwrite(output_path, canvas)
    print(f"Saved inference visualization to {output_path}")

if __name__ == "__main__":
    visualize_inference()
