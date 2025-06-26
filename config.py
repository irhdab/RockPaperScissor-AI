"""
Configuration file for Rock Paper Scissors AI
Centralized settings for all components
"""

import os

# Data Collection Settings
DATA_COLLECTION = {
    'capture_area': (150, 50, 450, 350),  # (x1, y1, x2, y2) for ROI
    'image_size': (300, 300),
    'supported_formats': ['.jpg', '.jpeg', '.png']
}

# Model Settings
MODEL = {
    'input_shape': (300, 300, 3),
    'num_classes': 3,
    'base_model': 'DenseNet121',
    'weights': 'imagenet',
    'dropout_rate': 0.5,
    'learning_rate': 0.001
}

# Training Settings
TRAINING = {
    'batch_size': 16,
    'epochs': 10,
    'validation_split': 0.2,
    'early_stopping_patience': 5,
    'model_save_path': 'model.h5',
    'architecture_save_path': 'model.json',
    'history_save_path': 'training_history.json'
}

# Game Settings
GAME = {
    'num_rounds': 3,
    'countdown_duration': 3,  # seconds
    'prediction_duration': 0.5,  # seconds
    'total_round_duration': 4,  # seconds
    'fps': 30
}

# Labels and Mappings
LABELS = {
    'rock': [1., 0., 0.],
    'paper': [0., 1., 0.],
    'scissor': [0., 0., 1.]
}

WIN_RULES = {
    'rock': 'scissor',
    'scissor': 'paper', 
    'paper': 'rock'
}

# File Paths
PATHS = {
    'model_json': 'model.json',
    'model_weights': 'model.h5',
    'training_history': 'training_history.json',
    'training_plot': 'training_history.png'
}

# UI Settings
UI = {
    'window_title': 'Rock Paper Scissors AI',
    'text_color': (250, 250, 0),  # Yellow
    'rectangle_color': (255, 255, 255),  # White
    'font_scale': 1,
    'thickness': 2
}

# Validation
def validate_config():
    """Validate configuration settings"""
    # Check if required directories exist
    required_dirs = ['rock', 'paper', 'scissor']
    
    # Validate model settings
    assert MODEL['num_classes'] == len(LABELS), "Number of classes must match number of labels"
    assert MODEL['input_shape'][2] == 3, "Input shape must have 3 channels (RGB)"
    
    # Validate game settings
    assert GAME['num_rounds'] > 0, "Number of rounds must be positive"
    assert GAME['countdown_duration'] > 0, "Countdown duration must be positive"
    
    return True

# Environment variables
def get_env_setting(key, default=None):
    """Get setting from environment variable with fallback to default"""
    return os.environ.get(key, default)

# GPU Settings
GPU_SETTINGS = {
    'use_gpu': get_env_setting('USE_GPU', 'True').lower() == 'true',
    'memory_growth': get_env_setting('GPU_MEMORY_GROWTH', 'True').lower() == 'true',
    'mixed_precision': get_env_setting('MIXED_PRECISION', 'False').lower() == 'true'
} 