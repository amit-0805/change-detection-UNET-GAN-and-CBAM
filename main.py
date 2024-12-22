import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
    parser.add_argument("--root_path", type=str, default="/workspace/NAS_MOUNT/", help="root path")
    parser.add_argument("--dataset_name", type=str, default="LEVIR-CD", help="name of the dataset")
    parser.add_argument("--save_name", type=str, default="levir", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    parser.add_argument("--sample_interval", type=int, default=2000, help="interval between sampling of images from generators")
    return parser.parse_args()

def sample_images(val_dataloader, generator, opt, device, batches_done):
    """Sample images function"""
    imgs = next(iter(val_dataloader))
    img_A = imgs["A"].to(device)
    img_B = imgs["B"].to(device)
    img_B_fake = generator(img_A, img_B)
    img_A = img_A[:, [2,1,0],:,:]
    img_B_fake = img_B_fake[:, [2,1,0],:,:]
    img_B = img_B[:, [2,1,0],:,:]
    img_sample = torch.cat((img_A.data, img_B_fake.data, img_B.data), -2)
    save_image(img_sample, "images/%s/%d.png" % (opt.save_name, batches_done), nrow=5, normalize=True)


def main():
    opt = parse_args()
    print(opt)

    # Create directories
    os.makedirs("images/%s" % opt.save_name, exist_ok=True)
    os.makedirs("saved_models/%s" % opt.save_name, exist_ok=True)

    # Set up device
    device = get_device()
    print(f"Using device: {device}")

    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()

    lambda_pixel = 100
    patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

    # Initialize generator and discriminator
    generator = GeneratorUNet_CBAM(in_channels=3).to(device)
    discriminator = Discriminator().to(device)

    # Move loss functions to device
    criterion_GAN = criterion_GAN.to(device)
    criterion_pixelwise = criterion_pixelwise.to(device)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Configure transforms
    transforms_aug = A.Compose([
        A.Resize(opt.img_height, opt.img_width),
        A.Normalize(), 
        ToTensorV2()
    ])

    transforms_ori = A.Compose([
        A.Resize(opt.img_height, opt.img_width),
        A.Normalize(), 
        ToTensorV2()
    ])

    # Configure data loaders
    dataloader = DataLoader(
        CDRL_Dataset(root_path=opt.root_path, dataset=opt.dataset_name, train_val='train', 
                    transforms_A=transforms_aug, transforms_B=transforms_ori),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    val_dataloader = DataLoader(
        CDRL_Dataset(root_path=opt.root_path, dataset=opt.dataset_name,  train_val='train', 
                    transforms_A=transforms_ori, transforms_B=transforms_ori),
        batch_size=10,
        shuffle=False,
        num_workers=1,
    )

    # ----------
    #  Training
    # ----------
    prev_time = time.time()

    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            # Configure input
            img_A = batch["A"].to(device)
            img_B = batch["B"].to(device)

            valid = torch.ones((img_A.size(0), *patch), device=device)
            fake = torch.zeros((img_A.size(0), *patch), device=device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            gener_output = generator(img_A, img_B)
            gener_output_pred = discriminator(gener_output, img_A)
            
            loss_GAN = criterion_GAN(gener_output_pred, valid)  
            loss_pixel = criterion_pixelwise(gener_output, img_A)
            
            loss_G = loss_GAN + lambda_pixel * loss_pixel
            loss_G.backward()
            
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            pred_real = discriminator(img_B, img_A)
            loss_real = criterion_GAN(pred_real, valid)

            B_pred_fake = discriminator(gener_output.detach(), img_A)
            loss_fake = criterion_GAN(B_pred_fake, fake)

            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_pixel.item(),
                    loss_GAN.item(),
                    time_left,
                )
            )

            if batches_done % opt.sample_interval == 0:
                sample_images(val_dataloader, generator, opt, device, batches_done)

        # Save models at end of epoch
        torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.save_name, epoch))
        torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.save_name, epoch))
        

if __name__ == "__main__":
    main()