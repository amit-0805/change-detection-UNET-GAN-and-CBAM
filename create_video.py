import cv2
import os
import numpy as np
from glob import glob
from tqdm import tqdm

def create_comparison_video(results_dir, output_path, fps=0.5):
    """
    Create a video from frame folders containing input frames and change detection mask.
    Each frame will be shown for 2 seconds.
    
    Args:
        results_dir: Base directory containing frame_XXXXXX folders
        output_path: Path where the output video will be saved
        fps: Frames per second for output video (default=0.5 for 2 seconds per frame)
    """
    # Get all frame folders sorted numerically
    frame_folders = glob(os.path.join(results_dir, 'frame_*'))
    if not frame_folders:
        raise ValueError(f"No frame folders found in {results_dir}")
    
    # Sort folders numerically
    frame_folders = sorted(frame_folders, 
                         key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    print(f"Found {len(frame_folders)} frame folders")
    print(f"First folder: {frame_folders[0]}")  # Debug print
    print(f"Each frame will be shown for {1/fps:.1f} seconds")
    
    # Read first frame to get dimensions
    first_frame_folder = frame_folders[0]
    first_t1 = cv2.imread(os.path.join(first_frame_folder, 'input_t1.png'))
    
    if first_t1 is None:
        print(f"Debug: Checking if file exists: {os.path.join(first_frame_folder, 'input_t1.png')}")
        print(f"Debug: Files in first folder: {os.listdir(first_frame_folder)}")
        raise ValueError(f"Could not read first frame from {os.path.join(first_frame_folder, 'input_t1.png')}")
    
    frame_height, frame_width = first_t1.shape[:2]
    
    # Calculate dimensions for the combined frame
    combined_width = frame_width * 3  # Three images side by side
    combined_height = frame_height
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (combined_width, combined_height)
    )
    
    print("Creating video...")
    for folder in tqdm(frame_folders):
        # Define the paths for the three images
        t1_path = os.path.join(folder, 'input_t1.png')
        t2_path = os.path.join(folder, 'input_t2.png')
        mask_path = os.path.join(folder, 'change_mask.png')
        
        if not all(os.path.exists(p) for p in [t1_path, t2_path, mask_path]):
            print(f"Warning: Missing files in {folder}")
            print(f"Debug: Files in folder: {os.listdir(folder)}")
            continue
            
        t1_frame = cv2.imread(t1_path)
        t2_frame = cv2.imread(t2_path)
        mask_frame = cv2.imread(mask_path)
        
        if any(img is None for img in [t1_frame, t2_frame, mask_frame]):
            print(f"Warning: Could not read images from {folder}")
            continue
            
        # Create combined frame
        combined_frame = np.hstack([t1_frame, t2_frame, mask_frame])
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined_frame, 'Past Frame', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(combined_frame, 'Present Frame', (frame_width + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(combined_frame, 'Change Mask', (frame_width * 2 + 10, 30), font, 1, (255, 255, 255), 2)
        
        # Add frame number
        frame_num = os.path.basename(folder).split('_')[1].split('.')[0]
        cv2.putText(combined_frame, f'Frame: {int(frame_num)}', (10, combined_height - 20), 
                    font, 0.8, (255, 255, 255), 2)
        
        # Write frame
        out.write(combined_frame)
    
    # Release video writer
    out.release()
    print(f"\nVideo saved to: {output_path}")
    print(f"Total video duration: {len(frame_folders)/fps:.1f} seconds")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create video from frame results')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing frame_XXXXXX folders')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path where the output video will be saved')
    parser.add_argument('--fps', type=float, default=0.5,
                        help='Frames per second for output video (default=0.5 for 2 seconds per frame)')
    
    args = parser.parse_args()
    
    create_comparison_video(args.results_dir, args.output_path, args.fps)