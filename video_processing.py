import cv2
import os
import numpy as np
from glob import glob
import os
from glob import glob
import cv2
import torch
from torch.utils.data import Dataset

def extract_frames_from_two_videos(video_path_past, video_path_present, output_dir, frame_interval=1):
    """
    Extract frames from two videos and save them in A and B directories.
    
    Args:
        video_path_past: Path to the past video file
        video_path_present: Path to the present video file
        output_dir: Base directory to save the extracted frames
        frame_interval: Extract every nth frame (default=1)
    """
    os.makedirs(os.path.join(output_dir, 'A'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'B'), exist_ok=True)
    
    # Open both video files
    cap_past = cv2.VideoCapture(video_path_past)
    cap_present = cv2.VideoCapture(video_path_present)
    
    if not cap_past.isOpened() or not cap_present.isOpened():
        raise ValueError("Error opening one or both video files")
    
    frame_count = 0
    
    while True:
        ret_past, frame_past = cap_past.read()
        ret_present, frame_present = cap_present.read()
        
        if not ret_past or not ret_present:
            break
            
        if frame_count % frame_interval == 0:
            # Save past frame to A directory
            cv2.imwrite(os.path.join(output_dir, 'A', f'frame_{frame_count:06d}.jpg'), frame_past)
            # Save present frame to B directory
            cv2.imwrite(os.path.join(output_dir, 'B', f'frame_{frame_count:06d}.jpg'), frame_present)
        
        frame_count += 1
    
    cap_past.release()
    cap_present.release()
    return frame_count

class VideoDataset(Dataset):
    def __init__(self, root_path, transforms=None):
        """
        Dataset for video frame change detection.
        
        Args:
            root_path: Path containing A (previous frames) and B (current frames) directories
            transforms: Albumentations transforms to apply to the images
        """
        self.root_path = root_path
        self.transforms = transforms
        self.files = sorted(glob(os.path.join(root_path, 'A', '*.*')))
        
    def __getitem__(self, index):
        name = os.path.basename(self.files[index])
        
        # Read previous (A) and current (B) frames
        img_A = cv2.imread(self.files[index], cv2.IMREAD_COLOR)
        img_B = cv2.imread(self.files[index].replace('/A/', '/B/'), cv2.IMREAD_COLOR)
        
        if self.transforms:
            transformed_A = self.transforms(image=img_A)
            transformed_B = self.transforms(image=img_B)
            img_A = transformed_A["image"]
            img_B = transformed_B["image"]
            
        return {"A": img_A, "B": img_B, "NAME": name}
    
    def __len__(self):
        return len(self.files)