# Video Change Detection AI 🎥

A deep learning-based system for detecting and visualizing changes between two videos of the same location taken at different times. The system uses a U-Net architecture enhanced with CBAM (Convolutional Block Attention Module) attention mechanism, trained on the LEVIR-CD dataset.

## Features 🌟

- Automated change detection between two video sequences
- Real-time visualization of detected changes
- Interactive web interface using Streamlit
- Support for multiple video formats (MP4, AVI, MOV)
- GPU acceleration support (CUDA and MPS)
- Comprehensive visualization with side-by-side comparison

## Installation 🛠️

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video-change-detection.git
cd video-change-detection
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download the resources:

- Pre-trained Models: [Download Models](https://drive.google.com/drive/folders/1GrBMsXJI27gabJS3hnWDA72aY_nKHFKR?usp=sharing)
- Dataset: [Download Dataset](https://drive.google.com/drive/folders/1IfhV4kgIRp1UmFAPQaWRYZ38VTo3Gg8E?usp=sharing)

Extract the downloaded files and place them in the following structure:
```
video-change-detection/
├── saved_models/
│   └── levir/
│       ├── generator_9.pth
│       └── discriminator_9.pth
├── dataset/
│   └── LEVIR-CD/
│       ├── train/
│       │   ├── A/  # Time 1 images
│       │   └── B/  # Time 2 images
│       └── val/
│           ├── A/  # Time 1 images
│           └── B/  # Time 2 images
```

## Dataset Details 📁

The dataset contains bi-temporal remote sensing images organized as follows:
- Training set: Pairs of images showing the same location at different times
- Validation set: Separate set of image pairs for model validation
- Images are organized in A (time 1) and B (time 2) directories
- Each image pair captures potential changes in the scene

## Training 🚂

To train the model from scratch:

```bash
python main.py \
    --root_path /path/to/dataset/LEVIR-CD \
    --dataset_name LEVIR-CD \
    --save_name levir \
    --n_epochs 10 \
    --batch_size 4 \
    --img_height 256 \
    --img_width 256
```

Key training parameters:
- `--n_epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--lr`: Learning rate (default: 0.0002)
- `--img_height` and `--img_width`: Input image dimensions

## Testing 🔍

To test the model on your own videos:

```bash
python test.py \
    --save_name levir \
    --video_path_past /path/to/past/video.mp4 \
    --video_path_present /path/to/present/video.mp4 \
    --frame_interval 1 \
    --save_visual
```

Parameters:
- `--video_path_past`: Path to the video from earlier time
- `--video_path_present`: Path to the recent video
- `--frame_interval`: Extract every nth frame (default: 1)
- `--save_visual`: Save visualization results

## Streamlit Web App 🌐

The project includes a user-friendly web interface built with Streamlit. To run the web app:

```bash
streamlit run app.py
```

The web app provides:
- Easy video upload interface
- Real-time processing status
- Side-by-side visualization of changes
- Interactive results viewer

### Using the Web App

1. Upload two videos of the same location taken at different times
2. Click the "Generate Change Detection" button
3. Wait for the processing to complete
4. View the results with side-by-side comparison

## Model Architecture 🏗️

The change detection system uses:
- Generator: U-Net architecture with CBAM attention
- Discriminator: PatchGAN discriminator
- Loss: Combination of adversarial loss and L1 loss

## Requirements 📋

- Python 3.8+
- PyTorch 1.8+
- Streamlit 1.0+
- OpenCV 4.5+
- Albumentations
- NumPy
- torchvision

## Results 📊

The model generates three types of outputs:
- Original frame from the first video (t1)
- Original frame from the second video (t2)
- Change mask highlighting detected changes between t1 and t2

The visualization shows:
- Red regions: Areas with significant changes
- Dark areas: Regions with no significant changes
- Brightness intensity: Indicates the magnitude of change
