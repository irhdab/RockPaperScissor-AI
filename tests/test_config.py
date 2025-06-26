"""
Unit tests for config.py
Tests configuration validation and settings
"""

import unittest
import sys
import os
import tempfile
import shutil

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class TestConfig(unittest.TestCase):
    """Test cases for configuration module"""

    def test_validate_config(self):
        """Test configuration validation"""
        # Should not raise any exceptions
        self.assertTrue(config.validate_config())

    def test_model_settings(self):
        """Test model configuration settings"""
        # Check model settings are valid
        self.assertEqual(config.MODEL['num_classes'], 3)
        self.assertEqual(config.MODEL['input_shape'], (300, 300, 3))
        self.assertEqual(config.MODEL['input_shape'][2], 3)  # RGB channels
        self.assertEqual(len(config.LABELS), config.MODEL['num_classes'])

    def test_game_settings(self):
        """Test game configuration settings"""
        # Check game settings are valid
        self.assertGreater(config.GAME['num_rounds'], 0)
        self.assertGreater(config.GAME['countdown_duration'], 0)
        self.assertGreater(config.GAME['prediction_duration'], 0)
        self.assertGreater(config.GAME['total_round_duration'], 0)

    def test_labels_consistency(self):
        """Test that labels are consistent"""
        # Check all labels have correct format
        for label, encoding in config.LABELS.items():
            self.assertEqual(len(encoding), 3)  # 3 classes
            self.assertEqual(sum(encoding), 1.0)  # One-hot encoding

    def test_win_rules(self):
        """Test win rules are complete and correct"""
        # Check all gestures have win rules
        for gesture in config.LABELS.keys():
            self.assertIn(gesture, config.WIN_RULES)
            self.assertIn(config.WIN_RULES[gesture], config.LABELS)

        # Test win rules logic
        self.assertEqual(config.WIN_RULES['rock'], 'scissor')
        self.assertEqual(config.WIN_RULES['paper'], 'rock')
        self.assertEqual(config.WIN_RULES['scissor'], 'paper')

    def test_data_collection_settings(self):
        """Test data collection settings"""
        # Check capture area is valid
        x1, y1, x2, y2 = config.DATA_COLLECTION['capture_area']
        self.assertLess(x1, x2)
        self.assertLess(y1, y2)

        # Check image size is valid
        width, height = config.DATA_COLLECTION['image_size']
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)

    def test_get_env_setting(self):
        """Test environment variable function"""
        # Test with default value
        result = config.get_env_setting('NONEXISTENT_VAR', 'default')
        self.assertEqual(result, 'default')

        # Test with existing environment variable
        os.environ['TEST_VAR'] = 'test_value'
        result = config.get_env_setting('TEST_VAR', 'default')
        self.assertEqual(result, 'test_value')
        del os.environ['TEST_VAR']

    def test_gpu_settings(self):
        """Test GPU settings"""
        # Check GPU settings are boolean
        self.assertIsInstance(config.GPU_SETTINGS['use_gpu'], bool)
        self.assertIsInstance(config.GPU_SETTINGS['memory_growth'], bool)
        self.assertIsInstance(config.GPU_SETTINGS['mixed_precision'], bool)


if __name__ == '__main__':
    unittest.main() 