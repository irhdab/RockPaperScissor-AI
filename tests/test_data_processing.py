"""
Unit tests for data processing functions
Tests image loading, preprocessing, and validation
"""

import unittest
import sys
import os
import tempfile
import shutil
import numpy as np
import cv2

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class TestDataProcessing(unittest.TestCase):
    """Test cases for data processing functions"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = os.path.join(self.temp_dir, 'test_data')
        os.makedirs(self.test_data_dir)
        
        # Create test directories
        for gesture in ['rock', 'paper', 'scissor']:
            os.makedirs(os.path.join(self.test_data_dir, gesture))

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)

    def test_image_preprocessing(self):
        """Test image preprocessing functions"""
        # Create a test image
        test_image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        
        # Test resizing
        resized = cv2.resize(test_image, config.DATA_COLLECTION['image_size'])
        self.assertEqual(resized.shape, config.DATA_COLLECTION['image_size'] + (3,))
        
        # Test cropping and resizing
        x1, y1, x2, y2 = config.DATA_COLLECTION['capture_area']
        cropped = test_image[y1:y2, x1:x2]
        resized_cropped = cv2.resize(cropped, config.DATA_COLLECTION['image_size'])
        self.assertEqual(resized_cropped.shape, config.DATA_COLLECTION['image_size'] + (3,))

    def test_data_validation(self):
        """Test data path validation"""
        # Test valid data path
        self.assertTrue(self.validate_data_path(self.test_data_dir))
        
        # Test invalid data path
        with self.assertRaises(FileNotFoundError):
            self.validate_data_path('/nonexistent/path')
        
        # Test missing gesture folders
        incomplete_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(incomplete_dir, 'rock'))
        with self.assertRaises(ValueError):
            self.validate_data_path(incomplete_dir)
        shutil.rmtree(incomplete_dir)

    def test_image_format_validation(self):
        """Test image format validation"""
        valid_formats = config.DATA_COLLECTION['supported_formats']
        
        # Test valid formats
        for fmt in valid_formats:
            self.assertTrue(self.is_valid_image_format(f'test{fmt}'))
        
        # Test invalid formats
        invalid_formats = ['.txt', '.pdf', '.doc', '.exe']
        for fmt in invalid_formats:
            self.assertFalse(self.is_valid_image_format(f'test{fmt}'))

    def test_label_encoding(self):
        """Test label encoding consistency"""
        # Test all labels are properly encoded
        for gesture, encoding in config.LABELS.items():
            # Should be numpy array
            self.assertIsInstance(encoding, list)
            
            # Should have correct length
            self.assertEqual(len(encoding), config.MODEL['num_classes'])
            
            # Should be one-hot encoded
            self.assertEqual(sum(encoding), 1.0)
            self.assertEqual(max(encoding), 1.0)
            self.assertEqual(min(encoding), 0.0)

    def test_data_augmentation(self):
        """Test data augmentation techniques"""
        # Create test image
        test_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        
        # Test horizontal flip
        flipped = cv2.flip(test_image, 1)
        self.assertEqual(flipped.shape, test_image.shape)
        
        # Test that flip is different from original
        self.assertFalse(np.array_equal(test_image, flipped))
        
        # Test that double flip returns original
        double_flipped = cv2.flip(flipped, 1)
        self.assertTrue(np.array_equal(test_image, double_flipped))

    def test_image_quality_validation(self):
        """Test image quality validation"""
        # Test valid image
        valid_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        self.assertTrue(self.is_valid_image(valid_image))
        
        # Test None image
        self.assertFalse(self.is_valid_image(None))
        
        # Test empty image
        empty_image = np.array([])
        self.assertFalse(self.is_valid_image(empty_image))
        
        # Test image with wrong dimensions
        wrong_dim_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        self.assertFalse(self.is_valid_image(wrong_dim_image))

    def validate_data_path(self, data_path):
        """Helper function to validate data path"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        
        required_folders = ['rock', 'paper', 'scissor']
        missing_folders = []
        
        for folder in required_folders:
            folder_path = os.path.join(data_path, folder)
            if not os.path.exists(folder_path):
                missing_folders.append(folder)
        
        if missing_folders:
            raise ValueError(f"Missing required folders: {missing_folders}")
        
        return True

    def is_valid_image_format(self, filename):
        """Helper function to check image format"""
        return any(filename.lower().endswith(fmt) for fmt in config.DATA_COLLECTION['supported_formats'])

    def is_valid_image(self, img):
        """Helper function to validate image"""
        if img is None:
            return False
        if img.size == 0:
            return False
        if len(img.shape) != 3 or img.shape[2] != 3:
            return False
        return True


if __name__ == '__main__':
    unittest.main() 