import numpy as np
import cv2
import os
import argparse
import matplotlib.pyplot as plt

def visualize_labels(event_dir, label_dir, frame_idx=0, sensor_w=346, sensor_h=260):
    """
    Overlays generated labels on event frames to verify alignment.
    """
    
    # Load Events (Accumulated frame)
    event_file = os.path.join(event_dir, f"frame_{frame_idx:06d}_cam2.npy")
    label_file = os.path.join(label_dir, f"frame_{frame_idx:06d}_label.npy")
    
    if not os.path.exists(event_file):
        print(f"Event file not found: {event_file}")
        return
    if not os.path.exists(label_file):
        print(f"Label file not found: {label_file}")
        return
        
    # Events are usually [H, W] or [2, H, W] - Let's check DHP19 output format
    # Based on previous exploration, it might be rasterized or raw.
    # checking file size: 32896 bytes. 346*260 = ~90k floats. 
    # 32896 bytes is small. 
    # Let's inspect content shape first.
    
    events = np.load(event_file)
    labels = np.load(label_file)
    
    print(f"Events Shape: {events.shape}")
    print(f"Labels Shape: {labels.shape}") # Should be [3, 13]
    
    # Create an image canvas
    canvas = np.zeros((sensor_h, sensor_w, 3), dtype=np.uint8)
    
    # If events are raw list [N, 4] -> Assume (t, x, y, p) or (x, y, t, p)
    # Usually DHP19 is (t, x, y, p)
    # Let's try to rasterize
    if len(events.shape) == 2 and events.shape[1] == 4:
        # Check values to guess columns
        # If col 1 is x, it should be < 346
        # If col 2 is y, it should be < 260
        # If col 0 is t, it is large
        
        # Simple heuristic or just assume standard x,y if coming from v2e
        # Actually v2e usually outputs .aedat or .h5
        # These .npy in custom might constitute the "accumulation"
        # Let's assume standard (t, x, y, p) as per DHP19
        
        # Let's inspect range to be safe in a print, but for now implement (t, x, y, p)
        # Note: If it's (x, y, t, p)
        
        # Rasterized.py indicates: x, y, t, p for columns 0, 1, 2, 3
        
        xs = events[:, 0].astype(int)
        ys = events[:, 1].astype(int)
        ts = events[:, 2]
        ps = events[:, 3].astype(int)
        
        # Bounds check
        mask = (xs >= 0) & (xs < sensor_w) & (ys >= 0) & (ys < sensor_h)
        xs = xs[mask]
        ys = ys[mask]
        ps = ps[mask]
        
        # Plot
        # Polarity 0 -> Red, 1 -> Green
        # Canvas: BGR
        
        # Vectorized plotting is hard with overwrites, simple loop is fine for N=2000
        canvas[ys, xs, 1] = 255 # Green for all events for visibility first
        
        # Split by polarity if desired
        # canvas[ys[ps==1], xs[ps==1], 1] = 255 # Green
        # canvas[ys[ps==0], xs[ps==0], 2] = 255 # Red (BGR -> 2 is Red)
        
    elif len(events.shape) == 2:
        # Gray image
        norm_events = cv2.normalize(events, None, 0, 255, cv2.NORM_MINMAX)
        canvas[:, :, 1] = norm_events.astype(np.uint8) 
    elif len(events.shape) == 3:
         # Assume channel first [C, H, W] or [H, W, C]
         if events.shape[0] == 2: # Polarity
             canvas[:, :, 0] = (events[0] * 255).astype(np.uint8) # Red
             canvas[:, :, 1] = (events[1] * 255).astype(np.uint8) # Green
         elif events.shape[2] == 2:
             canvas[:, :, 0] = (events[:,:,0] * 255).astype(np.uint8) # Red
             canvas[:, :, 1] = (events[:,:,1] * 255).astype(np.uint8) # Green
             
    # Draw Labels
    # Labels shape: [3, 13] -> (x, y, conf)
    num_joints = labels.shape[1]
    
    # DHP19 Joints: 
    # 0: Head, 1: R_Shoulder, 2: R_Elbow, 3: R_Wrist, 4: L_Shoulder, 5: L_Elbow, 6: L_Wrist
    connections = [
        (0, 1), (1, 2), (2, 3), # Right Arm
        (0, 4), (4, 5), (5, 6)  # Left Arm
    ]
    
    for i in range(num_joints):
        x = int(labels[0, i])
        y = int(labels[1, i])
        conf = labels[2, i]
        
        if conf > 0.1 and x > 0 and y > 0:
            cv2.circle(canvas, (x, y), 3, (0, 255, 255), -1) # Yellow joints
            cv2.putText(canvas, str(i), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Draw Bones
    for idx1, idx2 in connections:
        if idx1 < num_joints and idx2 < num_joints:
             x1, y1 = int(labels[0, idx1]), int(labels[1, idx1])
             x2, y2 = int(labels[0, idx2]), int(labels[1, idx2])
             c1 = labels[2, idx1]
             c2 = labels[2, idx2]
             
             if c1 > 0.1 and c2 > 0.1:
                 cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 1)

    output_file = "verify_output.png"
    cv2.imwrite(output_file, canvas)
    print(f"Saved visualization to {output_file}")

if __name__ == "__main__":
    # Hardcoded test path for verification
    base_dir = r"y:\BIC\DHP19EPC_dataset\custom\CARROT_TR_0"
    visualize_labels(base_dir, base_dir, frame_idx=100)
