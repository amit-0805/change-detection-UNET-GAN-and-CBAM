import streamlit as st
import os
import torch
from models import GeneratorUNet_CBAM, Discriminator
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from video_processing import extract_frames_from_two_videos, VideoDataset
from torch.utils.data import DataLoader
import tempfile
from create_video import create_comparison_video
from torchvision.utils import save_image
import shutil

# Page configuration
st.set_page_config(
    page_title="Video Change Detection AI",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .stApp {
            background-color: #f5f7f9;
        }
        .main > div {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3rem;
            background-color: #FF4B4B;
            color: white;
        }
        .stButton>button:hover {
            background-color: #FF3333;
        }
        .upload-section {
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .results-section {
            margin-top: 2rem;
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .info-box {
            background-color: #e1f5fe;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def process_videos(video_path_past, video_path_present, progress_bar, status_area):
    with tempfile.TemporaryDirectory() as temp_dir:
        progress_bar.progress(0.1)
        with status_area:
            st.info("📥 Extracting frames from videos...")
        
        try:
            num_frames = extract_frames_from_two_videos(
                video_path_past,
                video_path_present,
                temp_dir,
                frame_interval=1
            )
            with status_area:
                st.success(f"✅ Successfully extracted {num_frames} frame pairs")
        except Exception as e:
            with status_area:
                st.error(f"❌ Error during frame extraction: {str(e)}")
            return None
        
        progress_bar.progress(0.3)
        with status_area:
            st.info("🔧 Initializing AI model...")
        
        device = get_device()
        with status_area:
            st.success(f"💻 Using device: {device}")
        
        generator = GeneratorUNet_CBAM(in_channels=3).to(device)
        
        try:
            generator.load_state_dict(torch.load(
                "saved_models/levir/generator_9.pth",
                map_location=device
            ))
            generator.eval()
        except Exception as e:
            with status_area:
                st.error(f"❌ Error loading model weights: {str(e)}")
            return None
        
        progress_bar.progress(0.5)
        with status_area:
            st.info("🔄 Processing frames...")
        
        transforms_ = A.Compose([
            A.Resize(256, 256),
            A.Normalize(),
            ToTensorV2()
        ])
        
        val_dataset = VideoDataset(temp_dir, transforms=transforms_)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
        results_base_dir = os.path.join(temp_dir, 'results')
        os.makedirs(results_base_dir, exist_ok=True)
        
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                img_A = batch["A"].to(device)
                img_B = batch["B"].to(device)
                name = batch["NAME"]
                
                gener_output = generator(img_A, img_B)
                
                frame_dir = os.path.join(results_base_dir, f"frame_{i:06d}")
                os.makedirs(frame_dir, exist_ok=True)
                
                save_image(img_A.cpu(), os.path.join(frame_dir, 'input_t1.png'), normalize=True)
                save_image(img_B.cpu(), os.path.join(frame_dir, 'input_t2.png'), normalize=True)
                save_image(gener_output.cpu(), os.path.join(frame_dir, 'change_mask.png'), normalize=True)
                
                progress_bar.progress(0.5 + (0.4 * (i + 1) / len(val_dataloader)))
        
        with status_area:
            st.info("🎬 Creating final visualization...")
        output_video_path = os.path.join(temp_dir, 'output.mp4')
        
        try:
            create_comparison_video(results_base_dir, output_video_path, fps=0.5)
            
            with open(output_video_path, 'rb') as f:
                video_bytes = f.read()
            
            progress_bar.progress(1.0)
            return video_bytes
            
        except Exception as e:
            with status_area:
                st.error(f"❌ Error creating video: {str(e)}")
                st.write("📁 Debug: Contents of results directory:")
                for root, dirs, files in os.walk(results_base_dir):
                    st.write(f"Directory: {root}")
                    st.write(f"Files: {files}")
            return None

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=AI+Vision", width=150)
        st.title("About")
        st.markdown("""
        This AI-powered tool detects and visualizes changes between two videos 
        of the same location taken at different times.
        
        ### Applications
        - 🏗️ Construction monitoring
        - 🌳 Environmental change detection
        - 🏙️ Urban development tracking
        - 🔍 Security surveillance
        """)
        
        st.markdown("---")
        st.markdown("### Technical Details")
        with st.expander("Model Architecture"):
            st.markdown("""
            - Based on U-Net architecture
            - Enhanced with CBAM attention
            - Trained on LEVIR-CD dataset
            """)
    
    # Main content
    st.title("🎥 Video Change Detection AI")
    st.markdown("""
    Upload two videos of the same location taken at different times to detect and visualize changes.
    Our AI model will analyze the differences and generate a detailed change detection visualization.
    """)
    
    # Upload section
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📤 Past Video")
        video_past = st.file_uploader("Upload video from earlier time", type=['mp4', 'avi', 'mov'])
        if video_past:
            st.success("✅ Past video uploaded")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📤 Present Video")
        video_present = st.file_uploader("Upload recent video", type=['mp4', 'avi', 'mov'])
        if video_present:
            st.success("✅ Present video uploaded")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Guidelines
    with st.expander("📋 Guidelines for Best Results"):
        st.markdown("""
        1. **Video Requirements**
           - Same scene/location
           - Similar duration
           - Stable camera position
           - Good lighting conditions
           
        2. **Supported Formats**
           - MP4
           - AVI
           - MOV
           
        3. **Processing Time**
           - Depends on video length
           - Typically 2-5 minutes
           - Please be patient during processing
        """)
    
    # Process button
    if st.button('🚀 Generate Change Detection', disabled=not (video_past and video_present)):
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.subheader("🔄 Processing Status")
        
        progress_bar = st.progress(0)
        status_area = st.empty()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f1:
            f1.write(video_past.read())
            video_path_past = f1.name
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f2:
            f2.write(video_present.read())
            video_path_present = f2.name
        
        try:
            video_bytes = process_videos(video_path_past, video_path_present, progress_bar, status_area)
            
            if video_bytes:
                st.success("✨ Change detection completed successfully!")
                st.subheader("🎦 Results Visualization")
                st.video(video_bytes)
                
                st.markdown("""
                ### 📊 Visualization Guide
                - **Red regions**: Significant changes detected
                - **Dark areas**: No significant changes
                - **Brightness**: Indicates change intensity
                """)
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.warning("Please check that your videos are valid and try again.")
        
        finally:
            os.unlink(video_path_past)
            os.unlink(video_path_present)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()