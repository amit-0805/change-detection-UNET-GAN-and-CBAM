import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import os
from torchmetrics import JaccardIndex, Precision, Recall, F1Score, Accuracy

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# Import custom modules
from models import GeneratorUNet_CBAM, Discriminator
from video_processing import VideoDataset
from test import main as test_main
from create_video import create_comparison_video

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_model(model_path, model_type='generator'):
    """
    Load the generator or discriminator model
    """
    device = get_device()
    
    if model_type == 'generator':
        model = GeneratorUNet_CBAM(in_channels=3).to(device)
    else:
        model = Discriminator().to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def analyze_change_detection_metrics(val_dataloader, generator, discriminator):
    """
    Compute detailed metrics for change detection
    """
    device = get_device()
    criterion_GAN = torch.nn.MSELoss().to(device)
    criterion_pixelwise = torch.nn.L1Loss().to(device)
    lambda_pixel = 100
    
    # Initialize torchmetrics
    jaccard = JaccardIndex(task="binary").to(device)
    precision = Precision(task="binary").to(device)
    recall = Recall(task="binary").to(device)
    f1 = F1Score(task="binary").to(device)
    accuracy = Accuracy(task="binary").to(device)
    
    metrics = {
        'frame_names': [],
        'loss_G': [],
        'loss_pixel': [],
        'loss_GAN': [],
        'pixel_difference': [],
        'iou_score': [],
        'precision_score': [],
        'recall_score': [],
        'f1_score': [],
        'accuracy_score': []
    }
    
    with torch.no_grad():
        for batch in val_dataloader:
            img_A = batch["A"].to(device)
            img_B = batch["B"].to(device)
            name = batch["NAME"][0]
            
            valid = torch.ones((img_A.size(0), 1, 16, 16), device=device)
            
            gener_output = generator(img_A, img_B)
            gener_output_pred = discriminator(gener_output, img_A)
            
            # Calculate GAN losses
            loss_GAN = criterion_GAN(gener_output_pred, valid)
            loss_pixel = criterion_pixelwise(gener_output, img_A)
            loss_G = loss_GAN + lambda_pixel * loss_pixel
            
            # Calculate pixel difference
            pixel_diff = torch.mean(torch.abs(gener_output - img_A)).item()
            
            # Convert outputs to binary for metric calculation
            binary_threshold = 0.5
            binary_output = (gener_output > binary_threshold).float()
            binary_target = (img_A > binary_threshold).float()
            
            # Calculate additional metrics
            iou = jaccard(binary_output, binary_target)
            prec = precision(binary_output, binary_target)
            rec = recall(binary_output, binary_target)
            f1_metric = f1(binary_output, binary_target)
            acc = accuracy(binary_output, binary_target)
            
            # Store all metrics
            metrics['frame_names'].append(name)
            metrics['loss_G'].append(loss_G.item())
            metrics['loss_pixel'].append(loss_pixel.item())
            metrics['loss_GAN'].append(loss_GAN.item())
            metrics['pixel_difference'].append(pixel_diff)
            metrics['iou_score'].append(iou.item())
            metrics['precision_score'].append(prec.item())
            metrics['recall_score'].append(rec.item())
            metrics['f1_score'].append(f1_metric.item())
            metrics['accuracy_score'].append(acc.item())
    
    return pd.DataFrame(metrics)

def display_metrics_summary(metrics_df):
    """
    Display a summary of the metrics with formatted values
    """
    metrics_summary = {
        'Accuracy': metrics_df['accuracy_score'].mean(),
        'F1 Score': metrics_df['f1_score'].mean(),
        'IoU Score': metrics_df['iou_score'].mean(),
        'Precision': metrics_df['precision_score'].mean(),
        'Recall': metrics_df['recall_score'].mean()
    }
    
    # Create three columns for metric display
    col1, col2, col3 = st.columns(3)
    
    # Display metrics in a grid layout with larger text and formatting
    # with col1:
    #     st.metric("Accuracy", f"{metrics_summary['Accuracy']:.2%}")
    #     st.metric("IoU Score", f"{metrics_summary['IoU Score']:.2%}")
    
    # with col2:
    #     st.metric("F1 Score", f"{metrics_summary['F1 Score']:.2%}")
    #     st.metric("Precision", f"{metrics_summary['Precision']:.2%}")
    
    # with col3:
    #     st.metric("Recall", f"{metrics_summary['Recall']:.2%}")
    
    # return metrics_summary

def main():
    st.set_page_config(page_title="Change Detection Analysis", layout="wide")
    st.title("Change Detection Model Metrics & Visualization")
    
    # Add metric explanations
    with st.expander("Metric Explanations"):
        st.markdown("""
        - **IoU (Jaccard Index)**: Measures the overlap between predicted and actual changes
        - **Precision**: Ratio of correctly identified changes to total predicted changes
        - **Recall**: Ratio of correctly identified changes to actual changes
        - **F1 Score**: Harmonic mean of precision and recall, balancing both metrics
        - **Accuracy**: Overall correctness of change detection predictions
        """)
    
    # Sidebar Configuration
    st.sidebar.header("Model & Data Configuration")
    
    # Video Input
    past_video = st.sidebar.file_uploader("Upload Past Video", type=['mp4', 'avi'])
    present_video = st.sidebar.file_uploader("Upload Present Video", type=['mp4', 'avi'])
    
    # Model Configuration
    save_name = st.sidebar.selectbox("Dataset", ["levir", "other_datasets"])
    frame_interval = st.sidebar.slider("Frame Extraction Interval", 1, 10, 1)
    
    # Analysis Options
    st.sidebar.header("Analysis Options")
    show_metrics = st.sidebar.checkbox("Show Metrics Table")
    show_plots = st.sidebar.checkbox("Show Metrics Plots")
    
    if st.sidebar.button("Run Analysis"):
        if past_video and present_video:
            # Temporary file handling
            with open(os.path.join("/tmp", "past_video.mp4"), "wb") as f:
                f.write(past_video.getbuffer())
            with open(os.path.join("/tmp", "present_video.mp4"), "wb") as f:
                f.write(present_video.getbuffer())
            
            # Configure paths
            root_path = "/tmp/change_detection"
            os.makedirs(root_path, exist_ok=True)
            
            # Extract Frames
            st.write("Extracting Video Frames...")
            from video_processing import extract_frames_from_two_videos
            extract_frames_from_two_videos(
                "/tmp/past_video.mp4", 
                "/tmp/present_video.mp4", 
                root_path, 
                frame_interval
            )
            
            # Prepare Transformations
            transforms_ = A.Compose([
                A.Resize(256, 256),
                A.Normalize(), 
                ToTensorV2()
            ])
            
            # Create Dataset
            val_dataset = VideoDataset(root_path, transforms=transforms_)
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=1, shuffle=False
            )
            
            # Load Models
            generator = load_model(f"saved_models/{save_name}/generator_9.pth")
            discriminator = load_model(f"saved_models/{save_name}/discriminator_9.pth", 'discriminator')
            
            # Analyze Metrics
            metrics_df = analyze_change_detection_metrics(val_dataloader, generator, discriminator)
            
            # Display Metric Summary Cards
            st.subheader("Performance Metrics Summary")
            metrics_summary = display_metrics_summary(metrics_df)
            
            # Display Loss Summary
            st.subheader("Loss Metrics Summary")
            loss_summary = {
                'Generator Loss': metrics_df['loss_G'].mean(),
                'Pixel Loss': metrics_df['loss_pixel'].mean(),
                'GAN Loss': metrics_df['loss_GAN'].mean()
            }
            
            # Display loss metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Generator Loss", f"{loss_summary['Generator Loss']:.4f}")
            with col2:
                st.metric("Pixel Loss", f"{loss_summary['Pixel Loss']:.4f}")
            with col3:
                st.metric("GAN Loss", f"{loss_summary['GAN Loss']:.4f}")
            
            # Display Results
            if show_metrics:
                st.subheader("Detailed Metrics Table")
                st.dataframe(metrics_df)
            
            # [Previous imports and functions remain the same until the plotting section in main()]

            if show_plots:
                st.subheader("Metrics Visualization")
                
                # Add general explanation about the plots
                st.markdown("""
                Below are three different visualizations that help understand the model's performance. 
                In the box plots, the box shows the range where 50% of the values fall, the line inside 
                the box is the median, and the whiskers show the full range excluding outliers. Outliers 
                are shown as individual points.
                """)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Original loss metrics
                    st.markdown("""
                    ### Loss Metrics Distribution
                    This plot shows the distribution of different loss values:
                    - **Generator Loss (loss_G)**: Overall loss of the generator model
                    - **Pixel Loss (loss_pixel)**: How different the generated images are from targets at pixel level
                    - **GAN Loss (loss_GAN)**: How well the generator fools the discriminator
                    
                    Lower values indicate better performance.
                    """)
                    fig1, ax1 = plt.subplots()
                    metrics_df.boxplot(column=['loss_G', 'loss_pixel', 'loss_GAN'])
                    plt.title("Loss Metrics Distribution")
                    st.pyplot(fig1)
                
                # with col2:
                #     # New evaluation metrics
                #     st.markdown("""
                #     ### Evaluation Metrics Distribution
                #     This plot shows the distribution of performance metrics:
                #     - **IoU Score**: Overlap between predicted and actual changes (0-1, higher is better)
                #     - **Precision**: Accuracy of detected changes (0-1, higher is better)
                #     - **Recall**: Proportion of actual changes detected (0-1, higher is better)
                #     - **F1 Score**: Balance between precision and recall (0-1, higher is better)
                #     - **Accuracy**: Overall correct predictions (0-1, higher is better)
                #     """)
                #     fig2, ax2 = plt.subplots()
                #     metrics_df.boxplot(column=['iou_score', 'precision_score', 'recall_score', 'f1_score', 'accuracy_score'])
                #     plt.title("Evaluation Metrics Distribution")
                #     plt.xticks(rotation=45)
                #     st.pyplot(fig2)
                
                # Additional visualization for pixel difference
                st.markdown("""
                ### Pixel Difference Distribution
                This histogram shows how much the model's outputs differ from the input images at the pixel level:
                - The x-axis shows the magnitude of pixel differences
                - The y-axis shows how frequently each difference occurs
                - The blue curve (KDE) shows the overall distribution shape
                - A narrower distribution centered near zero indicates more consistent predictions
                """)
                fig3, ax3 = plt.subplots()
                sns.histplot(metrics_df['pixel_difference'], kde=True)
                plt.title("Pixel Difference Distribution")
                st.pyplot(fig3)
                
                # Statistical summary of pixel differences
                st.markdown("### Pixel Difference Statistics")
                pixel_stats = {
                    'Mean': metrics_df['pixel_difference'].mean(),
                    'Median': metrics_df['pixel_difference'].median(),
                    'Std Dev': metrics_df['pixel_difference'].std(),
                    'Min': metrics_df['pixel_difference'].min(),
                    'Max': metrics_df['pixel_difference'].max()
                }
                
                # Display pixel difference statistics in columns
                cols = st.columns(5)
                for col, (stat_name, value) in zip(cols, pixel_stats.items()):
                    with col:
                        st.metric(stat_name, f"{value:.4f}")
                
                # Add interpretation guide
                st.markdown("""
                #### How to Interpret These Results:
                
                1. **Performance Metrics (Second Plot)**:
                   - All metrics range from 0 to 1 (or 0% to 100%)
                   - Higher values indicate better performance
                   - Look for consistent high values across all metrics
                
                2. **Loss Values (First Plot)**:
                   - Lower values indicate better model performance
                   - High variance (large boxes) suggests inconsistent performance
                   - Many outliers may indicate problematic frames
                
                3. **Pixel Differences (Histogram)**:
                   - Shows how much the model's output differs from input
                   - Smaller differences (closer to 0) are generally better
                   - The shape tells you about prediction consistency
                
                #### Key Terms:
                - **Box Plot**: Shows data distribution through quartiles
                - **Outliers**: Points that fall outside the normal range
                - **Distribution**: How values are spread across their range
                - **KDE (Kernel Density Estimation)**: Smooth curve showing value distribution
                """)

# [Rest of the code remains the same]
                
                # Display average pixel difference
                st.metric("Average Pixel Difference", f"{metrics_df['pixel_difference'].mean():.4f}")
        
        else:
            st.error("Please upload both past and present videos")

if __name__ == "__main__":
    main()