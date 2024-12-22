import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from video_processing import *

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def pixel_visual(gener_output_, A_ori_, name, save_name):
    # Move tensors to CPU for visualization processing
    gener_output = gener_output_.cpu().clone().detach().squeeze()
    A_ori = A_ori_.cpu().clone().detach().squeeze()
    
    pixel_loss = to_pil_image(torch.abs(gener_output - A_ori))
    trans = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()])
    pixel_loss = trans(pixel_loss)

    thre_num = 0.7
    threshold = nn.Threshold(thre_num, 0.)
    pixel_loss = threshold(pixel_loss)
    save_image(pixel_loss, f'pixel_img/{save_name}/{str(name[0])}')
    save_image(gener_output.flip(-3), f'gener_img/{save_name}/{str(name[0])}', normalize=True)

def visualize_change_detection(img_A, img_B, gener_output, save_path):
    # Convert tensors to CPU and detach for visualization
    img_A = img_A.cpu().detach()
    img_B = img_B.cpu().detach()
    gener_output = gener_output.cpu().detach()

    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save individual images
    save_image(img_A, f'{save_path}/input_t1.png', normalize=True)
    save_image(img_B, f'{save_path}/input_t2.png', normalize=True)
    save_image(gener_output, f'{save_path}/change_mask.png', normalize=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="/workspace/NAS_MOUNT/", help="root path")
    parser.add_argument("--dataset_name", type=str, default="LEVIR-CD", help="name of the dataset")
    parser.add_argument("--save_name", type=str, default="levir", help="name of the dataset")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    parser.add_argument('--save_visual', action='store_true', help='save pixel visualization map')
    parser.add_argument("--video_path_past", type=str, required=False, help="path to past video file")
    parser.add_argument("--video_path_present", type=str, required=False, help="path to present video file")
    parser.add_argument("--frame_interval", type=int, default=1, help="extract every nth frame from videos")
    opt = parser.parse_args()
    print(opt)
    
    # Print directory structure before processing
    print("\nChecking input paths...")
    print(f"Video path (past): {opt.video_path_past} (exists: {os.path.exists(opt.video_path_past)})")
    print(f"Video path (present): {opt.video_path_present} (exists: {os.path.exists(opt.video_path_present)})")
    print(f"Root path: {opt.root_path} (exists: {os.path.exists(opt.root_path)})")
    
    # Create output directories if they don't exist
    os.makedirs(opt.root_path, exist_ok=True)
    os.makedirs(os.path.join(opt.root_path, 'A'), exist_ok=True)
    os.makedirs(os.path.join(opt.root_path, 'B'), exist_ok=True)
    
    # Check if frames already exist
    frames_a = glob(os.path.join(opt.root_path, 'A', '*.*'))
    frames_b = glob(os.path.join(opt.root_path, 'B', '*.*'))
    print(f"\nFound {len(frames_a)} frames in directory A")
    print(f"Found {len(frames_b)} frames in directory B")
    
    if len(frames_a) == 0 or len(frames_b) == 0:
        print("\nExtracting video frames...")
        try:
            num_frames = extract_frames_from_two_videos(
                opt.video_path_past,
                opt.video_path_present,
                opt.root_path,
                opt.frame_interval
            )
            print(f"Successfully extracted {num_frames} frame pairs")
            
            # Recount frames after extraction
            frames_a = glob(os.path.join(opt.root_path, 'A', '*.*'))
            frames_b = glob(os.path.join(opt.root_path, 'B', '*.*'))
            print(f"After extraction: Found {len(frames_a)} frames in directory A")
            print(f"After extraction: Found {len(frames_b)} frames in directory B")
            
            if len(frames_a) == 0 or len(frames_b) == 0:
                raise ValueError("No frames were extracted from the videos")
                
        except Exception as e:
            print(f"Error during frame extraction: {str(e)}")
            raise

    # Set up device
    device = get_device()
    print(f"\nUsing device: {device}")

    # Create output directories
    os.makedirs(f'pixel_img/{opt.save_name}', exist_ok=True)
    os.makedirs(f'gener_img/{opt.save_name}', exist_ok=True)
    os.makedirs(f'results/{opt.save_name}', exist_ok=True)

    # Initialize models and move to device
    print("\nInitializing models...")
    generator = GeneratorUNet_CBAM(in_channels=3).to(device)
    discriminator = Discriminator().to(device)

    # Load model weights
    print("Loading model weights...")
    generator.load_state_dict(torch.load(
        f"saved_models/{opt.save_name}/generator_9.pth",
        map_location=device
    ))
    discriminator.load_state_dict(torch.load(
        f"saved_models/{opt.save_name}/discriminator_9.pth",
        map_location=device
    ))

    transforms_ = A.Compose([
        A.Resize(opt.img_height, opt.img_width),
        A.Normalize(), 
        ToTensorV2()
    ])

    print("\nInitializing dataset...")
    val_dataset = VideoDataset(opt.root_path, transforms=transforms_)
    print(f"Dataset size: {len(val_dataset)} samples")
    
    if len(val_dataset) == 0:
        raise ValueError("Dataset is empty! No images found to process.")
        
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    # Rest of your processing code...
    generator.eval()
    discriminator.eval()
    
    criterion_GAN = torch.nn.MSELoss().to(device)
    criterion_pixelwise = torch.nn.L1Loss().to(device)
    lambda_pixel = 100
    patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)
    
    prev_time = time.time()
    loss_G_total = 0
    total_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            img_A = batch["A"].to(device)
            img_B = batch["B"].to(device)
            name = batch["NAME"]

            valid = torch.ones((img_A.size(0), *patch), device=device)

            gener_output = generator(img_A, img_B)
            gener_output_pred = discriminator(gener_output, img_A)
            
            if opt.save_visual:
                visualize_change_detection(
                    img_A,
                    img_B,
                    gener_output,
                    f'results/{opt.save_name}/{str(name[0])}'
                )
                
            loss_GAN = criterion_GAN(gener_output_pred, valid)  
            loss_pixel = criterion_pixelwise(gener_output, img_B)
            loss_G = loss_GAN + lambda_pixel * loss_pixel

            print('-----------------------------------------------------------------------------')
            print('name : ', name[0])
            print('loss_G : ', round(loss_G.item(), 4))
            loss_G_total += loss_G
            total_batches += 1
            
        if total_batches > 0:
            print('----------------------------total------------------------------')
            print('loss_G_total : ', round((loss_G_total/total_batches).item(), 4))
        else:
            print('No batches were processed!')

if __name__ == '__main__':
    main()