
import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
from tqdm import tqdm

def generate_labels(video_path, output_dir, sensor_size_w=346, sensor_size_h=260):
    """
    Generates pose labels from a video using MediaPipe and saves them in DHP19 format.
    Output shape: [3, 13] (x, y, confidence)
    
    Mapping MediaPipe (33 joints) to DHP19 (13 joints):
    0: Head (Use Nose)
    1: R_Shoulder -> 12
    2: R_Elbow -> 14
    3: R_Wrist -> 16
    4: L_Shoulder -> 11
    5: L_Elbow -> 13
    6: L_Wrist -> 15
    7-12: Legs/Hips (ignored/filled with 0 for now)
    """
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Check output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=frame_count, desc="Generating Labels")
    
    frame_idx = 0
    w_scale = sensor_size_w # Events are in this range
    h_scale = sensor_size_h 
    
    # Original video dimensions might differ, MP returns normalized [0, 1]
    # We map Normalized [0, 1] -> Sensor Size directly
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        # Initialize [3, 13] container
        # Rows: 0=x, 1=y, 2=confidence
        label_data = np.zeros((3, 13), dtype=np.float32)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Map joints
            # Key: DHP_Index, Value: MP_Index
            # DHP19 Joints: 
            # 0: Head, 1: R_Shoulder, 2: R_Elbow, 3: R_Wrist, 4: L_Shoulder, 5: L_Elbow, 6: L_Wrist
            # MP Joints:
            # 0: Nose (Head)
            # 12: R_Shoulder
            # 14: R_Elbow
            # 16: R_Wrist
            # 11: L_Shoulder
            # 13: L_Elbow
            # 15: L_Wrist
            
            joint_map = {
                0: 0, 
                1: 12, 
                2: 14, 
                3: 16, 
                4: 11,
                5: 13, 
                6: 15  
            }
            
            for dhp_idx, mp_idx in joint_map.items():
                lm = landmarks[mp_idx]
                
                # Check visibility
                if lm.visibility > 0.5:
                    label_data[0, dhp_idx] = lm.x * w_scale
                    label_data[1, dhp_idx] = lm.y * h_scale
                    label_data[2, dhp_idx] = 1.0 # Mask/Confidence
        
        # Save as .npy
        # Filename: frame_000000_label.npy matches frame_000000_cam2.npy
        filename = f"frame_{frame_idx:06d}_label.npy"
        save_path = os.path.join(output_dir, filename)
        np.save(save_path, label_data)
        
        frame_idx += 1
        pbar.update(1)
        
    cap.release()
    pbar.close()
    print(f"Generated labels for {frame_idx} frames in {output_dir}")

def process_dataset(moca_root, custom_root, sensor_w, sensor_h):
    """
    Walks through the MoCA training directory, finds all 'dvs-video.avi' files,
    and generates labels in the corresponding custom dataset directory.
    """
    moca_training_dir = os.path.join(moca_root, "data", "video", "training")
    
    if not os.path.exists(moca_training_dir):
        print(f"Error: MoCA training directory not found at {moca_training_dir}")
        return

    # Find all subdirectories in MoCA training
    subdirs = [d for d in os.listdir(moca_training_dir) if os.path.isdir(os.path.join(moca_training_dir, d))]
    
    print(f"Found {len(subdirs)} potential directories to process.")
    
    for subdir in subdirs:
        video_path = os.path.join(moca_training_dir, subdir, "dvs-video.avi")
        output_dir = os.path.join(custom_root, subdir)
        
        if os.path.exists(video_path):
            print(f"\nProcessing {subdir}...")
            # Create output dir if it doesn't exist (it should for events, but good to be safe)
            generate_labels(video_path, output_dir, sensor_w, sensor_h)
        else:
            print(f"Skipping {subdir}: 'dvs-video.avi' not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--moca_root', type=str, default=r'Y:\BIC\Dataset\MoCA\MoCA', help='Root of MoCA dataset')
    parser.add_argument('--custom_root', type=str, default=r'y:\BIC\DHP19EPC_dataset\custom', help='Root of Custom dataset (DHP19 format)')
    parser.add_argument('--width', type=int, default=346, help='Sensor width')
    parser.add_argument('--height', type=int, default=260, help='Sensor height')
    
    args = parser.parse_args()
    
    process_dataset(args.moca_root, args.custom_root, args.width, args.height)
