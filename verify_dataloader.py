import argparse
import sys
import os
import torch
import numpy as np

# Add project root to sys.path
sys.path.append(os.getcwd())

from dataset.DHP19EPC import DHP19EPC

class Args:
    def __init__(self):
        self.num_points = 1024
        self.label = 'last'
        self.sensor_sizeW = 346
        self.sensor_sizeH = 260
        self.sigma = 8

def test_dataloader():
    args = Args()
    
    # Path to Custom Dataset
    # We generated labels in DHP19EPC_dataset/custom/TRANSPORTING_TR_2
    root_data = r"y:\BIC\DHP19EPC_dataset\custom\TRANSPORTING_TR_2"
    root_label = r"y:\BIC\DHP19EPC_dataset\custom\TRANSPORTING_TR_2"
    
    print(f"Testing DHP19EPC with root: {root_data}")
    
    # Instantiate dataset
    # We pass root_label_dir same as data dir because we saved labels there
    dataset = DHP19EPC(args, root_data_dir=root_data, root_label_dir=root_label, Test3D=False)
    
    print(f"Dataset Length: {len(dataset)}")
    
    if len(dataset) == 0:
        print("Error: Dataset is empty.")
        return

    # Try loading item 747 which we know has labels
    try:
        data, x, y, weight = dataset[747]
        print("Successfully loaded item 747")
        print(f"Data Shape: {data.shape}") # Should be [2, 346, 260] or similar if processed? 
        # RasEventCloud_preprocess returns [N, 5] (x, y, t, p, cnt)? No, wait.
        # RasEventCloud_preprocess:
        # data = EventCloudDHP.convert(data).numpy()[:, 1:] # [x, y, t, p, cnt]
        # It calls EventCloudDHP...
        
        print(f"Label X Shape: {x.shape}")
        print(f"Label Y Shape: {y.shape}")
        
        # Check label content non-zero
        if np.max(x) == 0 and np.max(y) == 0:
            print("Warning: Label seems to be all zeros.")
        else:
            print("Label contains valid data.")
            
    except Exception as e:
        print(f"Error loading item: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataloader()
