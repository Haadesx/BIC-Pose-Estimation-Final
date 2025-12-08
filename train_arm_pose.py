
import sys
import os
import glob
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add current directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from dataset.DHP19EPC import DHP19EPC
    from models.PointNet import Pose_PointNet
except ImportError:
    # Fallback if running from root
    from EventPointPose.dataset.DHP19EPC import DHP19EPC
    from EventPointPose.models.PointNet import Pose_PointNet

class CustomDHP19EPC(DHP19EPC):
    def __init__(self, args, root_data_dir, root_label_dir, subset_frames=None):
        # Allow passing args or creating a dummy object if needed
        self.root_data_dir = root_data_dir
        self.root_label_dir = root_label_dir
        self.Test3D = False # We are training
        self.sample_point_num = args.num_points
        self.label = args.label
        self.sx = args.sensor_sizeW
        self.sy = args.sensor_sizeH
        self.num_joints = args.num_joints
        self.sigma = getattr(args, 'sigma', 8)
        
        # Override initial loading logic
        if subset_frames:
             self.dvs_frames = subset_frames
        else:
             # Glob all cam2 files in custom/dvs-video
             # We look for *cam2.npy files to match the filtering logic
             # Glob all cam2 files in custom (recursive)
             search_path = os.path.join(root_data_dir, "**", "*cam2.npy")
             self.dvs_frames = sorted(glob.glob(search_path, recursive=True))
        
        print(f"Custom Dataset initialized with {len(self.dvs_frames)} frames.")
        print(f"Data Source: {root_data_dir}")
        print(f"Label Source: {root_label_dir}")

def train_model(args, root_data_dir, root_label_dir, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Initialize Dataset
    dataset = CustomDHP19EPC(args, root_data_dir, root_label_dir)
    
    if len(dataset) == 0:
        print("Error: No data found. Exiting.")
        return

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True) # num_workers=0 for Windows safety
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

    # Initialize Model
    print(f"Initializing PointNet with num_joints={args.num_joints}")
    model = Pose_PointNet(args).to(device)
    
    # Loss and Optimizer
    # Loss and Optimizer
    criterion = nn.MSELoss(reduction='none').to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        # Use tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for i, (data, x, y, weight) in enumerate(pbar):
            # data: [B, N, C]
            # PointNet expects [B, C, N]
            
            data = data.float().to(device) 
            target_x = x.float().to(device)
            target_y = y.float().to(device)
            
            # Transpose data to [B, C, N]
            # Data from DHP19EPC is [B, N, 5] (x, y, t, p, event_cnt)
            data = data.transpose(2, 1) 
            
            optimizer.zero_grad()
            pred_x, pred_y = model(data)
            
            # Loss
            # Loss
            # Weight shape: [B, J, 1]. pred_x shape: [B, J, W]. Broadcast mask.
            mask = weight.float().to(device)
            loss_x = (criterion(pred_x, target_x) * mask).mean()
            loss_y = (criterion(pred_y, target_y) * mask).mean()
            loss = loss_x + loss_y
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        # Validation (Simple Loss Check)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
             for i, (data, x, y, weight) in enumerate(val_loader):
                 data = data.float().to(device).transpose(2, 1)
                 target_x = x.float().to(device)
                 target_y = y.float().to(device)
                 
                 pred_x, pred_y = model(data)
                 mask = weight.float().to(device)
                 loss = (criterion(pred_x, target_x) * mask).mean() + (criterion(pred_y, target_y) * mask).mean()
                 val_loss += loss.item()
                 
        avg_val_loss = val_loss / len(val_loader)
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} complete. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Save checkpoint
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(save_path, f"pointnet_epoch_{epoch}.pth"))
        print(f"Model saved to {save_path}")

    # Plot Loss Curve
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    print(f"Loss curve saved to {os.path.join(save_path, 'loss_curve.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_joints', type=int, default=6, help='Number of joints to predict (default: 6 for arm pose)')
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--sensor_sizeW', type=int, default=346)
    parser.add_argument('--sensor_sizeH', type=int, default=260)
    parser.add_argument('--label', type=str, default='mean', help="Label type: 'mean' or 'last'. For custom loaded data we handle this in dataset class.")
    parser.add_argument('--sigma', type=float, default=8)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    
    args = parser.parse_args()
    
    # Paths (Hardcoded for current user environment as per discovery)
    # Paths (Hardcoded for current user environment as per discovery)
    ROOT_DATA = r'y:\BIC\DHP19EPC_dataset\custom'
    ROOT_LABEL = r'y:\BIC\DHP19EPC_dataset\custom' # Unused now as logic is relative, but keeping valid path
    SAVE_PATH = r'y:\BIC\EventPointPose\checkpoints'
    
    if not os.path.exists(SAVE_PATH):
        try:
            os.makedirs(SAVE_PATH)
        except Exception as e:
            print(f"Could not create save path: {e}")
        
    train_model(args, ROOT_DATA, ROOT_LABEL, SAVE_PATH)
