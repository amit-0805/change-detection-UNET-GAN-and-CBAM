import unittest
import torch
import os
import tempfile
import numpy as np
from models import GeneratorUNet_CBAM
from video_processing import extract_frames_from_two_videos, VideoDataset
import cv2
from torchmetrics import JaccardIndex, Precision, Recall, F1Score
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import pytest

# Global dictionary to store metrics
METRICS = {}

@pytest.fixture(autouse=True)
def run_around_tests():
    yield
    if METRICS:  # Only print if metrics have been collected
        print("\n=== Final Metrics Summary ===")
        for metric_name, value in METRICS.items():
            print(f"{metric_name}: {value:.4f}")
        print("==========================")

class TestVideoChangeDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up device
        try:
            if torch.cuda.is_available():
                cls.device = torch.device("cuda")
            else:
                cls.device = torch.device("cpu")
        except:
            cls.device = torch.device("cpu")
            
        print(f"\nUsing device: {cls.device}")
        
        # Initialize model
        cls.generator = GeneratorUNet_CBAM(in_channels=3).to(cls.device)
        
        # Set up transforms
        cls.transforms = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_model_initialization(self):
        """Test if model initializes correctly"""
        try:
            self.assertIsInstance(self.generator, GeneratorUNet_CBAM)
            self.assertTrue(hasattr(self.generator, 'down1'))
            print("\nModel initialization test passed")
        except Exception as e:
            print(f"\nModel initialization test failed: {str(e)}")
            raise

    def test_model_output_shape(self):
        """Test if model produces correct output shape"""
        try:
            test_input1 = torch.randn(1, 3, 256, 256).to(self.device)
            test_input2 = torch.randn(1, 3, 256, 256).to(self.device)
            with torch.no_grad():
                output = self.generator(test_input1, test_input2)
            self.assertEqual(output.shape, (1, 3, 256, 256))
            METRICS['output_channels'] = float(output.shape[1])
            print("\nModel output shape test passed")
        except Exception as e:
            print(f"\nModel output shape test failed: {str(e)}")
            raise

    def test_frame_extraction(self):
        """Test frame extraction functionality"""
        try:
            video1_path = os.path.join(self.temp_dir, "video1.mp4")
            video2_path = os.path.join(self.temp_dir, "video2.mp4")
            
            self._create_dummy_video(video1_path)
            self._create_dummy_video(video2_path)
            
            frames_t1_dir = os.path.join(self.temp_dir, "frames_t1")
            frames_t2_dir = os.path.join(self.temp_dir, "frames_t2")
            
            os.makedirs(frames_t1_dir, exist_ok=True)
            os.makedirs(frames_t2_dir, exist_ok=True)
            
            num_frames = extract_frames_from_two_videos(
                video1_path,
                video2_path,
                self.temp_dir,
                frame_interval=1
            )
            
            self.assertTrue(num_frames > 0)
            self.assertTrue(os.path.exists(frames_t1_dir))
            self.assertTrue(os.path.exists(frames_t2_dir))
            METRICS['num_frames_extracted'] = float(num_frames)
            print(f"\nFrame extraction test passed (Extracted {num_frames} frames)")
        except Exception as e:
            print(f"\nFrame extraction test failed: {str(e)}")
            raise

    def test_dataset_loading(self):
        """Test if dataset loads correctly"""
        try:
            frames_a_dir = os.path.join(self.temp_dir, "A")
            frames_b_dir = os.path.join(self.temp_dir, "B")
            os.makedirs(frames_a_dir, exist_ok=True)
            os.makedirs(frames_b_dir, exist_ok=True)
            
            dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(frames_a_dir, "frame_000000.png"), dummy_image)
            cv2.imwrite(os.path.join(frames_b_dir, "frame_000000.png"), dummy_image)
            
            dataset = VideoDataset(
                root_path=self.temp_dir,
                transforms=self.transforms
            )
            
            self.assertEqual(len(dataset), 1)
            sample = dataset[0]
            self.assertIn("A", sample)
            self.assertIn("B", sample)
            self.assertEqual(sample["A"].shape, (3, 256, 256))
            METRICS['dataset_size'] = float(len(dataset))
            print("\nDataset loading test passed")
        except Exception as e:
            print(f"\nDataset loading test failed: {str(e)}")
            raise

    def test_model_metrics(self):
        """Test if model metrics can be computed correctly"""
        try:
            # Create dummy predictions and targets
            pred = torch.randint(0, 2, (1, 3, 256, 256), dtype=torch.float32).to(self.device)
            target = torch.randint(0, 2, (1, 3, 256, 256), dtype=torch.float32).to(self.device)
            
            # Initialize metrics
            jaccard = JaccardIndex(task='multiclass', num_classes=2).to(self.device)
            precision = Precision(task='multiclass', num_classes=2).to(self.device)
            recall = Recall(task='multiclass', num_classes=2).to(self.device)
            f1 = F1Score(task='multiclass', num_classes=2).to(self.device)
            
            # Compute metrics
            j_score = jaccard(pred, target)
            p_score = precision(pred, target)
            r_score = recall(pred, target)
            f1_score = f1(pred, target)
            
            # Store metrics
            METRICS['jaccard_score'] = j_score.item()
            METRICS['precision_score'] = p_score.item()
            METRICS['recall_score'] = r_score.item()
            METRICS['f1_score'] = f1_score.item()
            
            print("\nMetric Results:")
            print(f"Jaccard Score: {j_score.item():.4f}")
            print(f"Precision Score: {p_score.item():.4f}")
            print(f"Recall Score: {r_score.item():.4f}")
            print(f"F1 Score: {f1_score.item():.4f}")
            
            print("Model metrics test passed")
        except Exception as e:
            print(f"\nModel metrics test failed: {str(e)}")
            raise

    def _create_dummy_video(self, path, frames=10, size=(100, 100)):
        """Helper method to create dummy video for testing"""
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(path, fourcc, 1.0, size)
            
            for _ in range(frames):
                frame = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
                out.write(frame)
            out.release()
        except Exception as e:
            print(f"\nError creating dummy video: {str(e)}")
            raise

    def test_dataloader_creation(self):
        """Test if DataLoader works correctly with the dataset"""
        try:
            frames_a_dir = os.path.join(self.temp_dir, "A")
            frames_b_dir = os.path.join(self.temp_dir, "B")
            os.makedirs(frames_a_dir, exist_ok=True)
            os.makedirs(frames_b_dir, exist_ok=True)
            
            for i in range(3):
                dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                cv2.imwrite(os.path.join(frames_a_dir, f"frame_{i:06d}.png"), dummy_image)
                cv2.imwrite(os.path.join(frames_b_dir, f"frame_{i:06d}.png"), dummy_image)
            
            dataset = VideoDataset(
                root_path=self.temp_dir,
                transforms=self.transforms
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=True,
                num_workers=0
            )
            
            batch = next(iter(dataloader))
            self.assertIn("A", batch)
            self.assertIn("B", batch)
            self.assertEqual(batch["A"].shape[0], 2)
            METRICS['dataloader_batch_size'] = float(batch["A"].shape[0])
            print("\nDataLoader creation test passed")
        except Exception as e:
            print(f"\nDataLoader creation test failed: {str(e)}")
            raise

if __name__ == '__main__':
    pytest.main([__file__, '-v'])