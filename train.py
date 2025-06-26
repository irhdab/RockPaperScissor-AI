# -*- coding: utf-8 -*-
"""
Rock Paper Scissors AI Training Script
Improved version with better error handling and cross-platform compatibility
"""

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io
import os
import pickle
import sys
import json
from pathlib import Path

def validate_data_path(data_path):
    """Validate that the data path exists and contains required folders"""
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

def load_and_preprocess_data(data_path):
    """Load and preprocess the training data"""
    print("Loading and preprocessing data...")
    
    shape_to_label = {
        'rock': np.array([1., 0., 0.]),
        'paper': np.array([0., 1., 0.]),
        'scissor': np.array([0., 0., 1.])
    }
    
    img_data = []
    labels = []
    
    for folder in ['rock', 'paper', 'scissor']:
        if folder not in shape_to_label:
            continue
            
        folder_path = os.path.join(data_path, folder)
        print(f"Processing {folder} images...")
        
        lb = shape_to_label[folder]
        count = 0
        
        for image_file in os.listdir(folder_path):
            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            image_path = os.path.join(folder_path, image_file)
            
            try:
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Warning: Could not read image {image_path}")
                    continue
                
                # Original image
                img_data.append([img, lb])
                
                # Horizontally flipped image
                img_data.append([cv2.flip(img, 1), lb])
                
                # Cropped and resized image
                if img.shape[0] > 250 and img.shape[1] > 250:
                    cropped = cv2.resize(img[50:250, 50:250], (300, 300))
                    img_data.append([cropped, lb])
                else:
                    # If image is too small, just resize
                    resized = cv2.resize(img, (300, 300))
                    img_data.append([resized, lb])
                
                count += 3
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        print(f"Processed {count} images for {folder}")
    
    if not img_data:
        raise ValueError("No valid images found in the data directory")
    
    # Shuffle the data
    np.random.shuffle(img_data)
    
    # Separate images and labels
    img_data, labels = zip(*img_data)
    
    img_data = np.array(img_data)
    labels = np.array(labels)
    
    print(f"Total images loaded: {len(img_data)}")
    print(f"Data shape: {img_data.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return img_data, labels

def create_model():
    """Create and compile the model"""
    print("Creating model...")
    
    # Use DenseNet121 as base model
    base_model = tf.keras.applications.DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=(300, 300, 3)
    )
    
    # Freeze the base model initially
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, img_data, labels, epochs=10, batch_size=16):
    """Train the model with callbacks"""
    print("Training model...")
    
    # Create callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'model.h5',
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto'
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        x=img_data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    return history

def save_model(model, history):
    """Save the model and training history"""
    print("Saving model...")
    
    # Save model architecture
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    
    # Save model weights (already saved by checkpoint)
    print("Model saved as model.json and model.h5")
    
    # Save training history
    with open("training_history.json", "w") as f:
        json.dump(history.history, f)
    
    print("Training history saved as training_history.json")

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def main():
    """Main training function"""
    if len(sys.argv) != 2:
        print("Usage: python train.py <path_to_images>")
        print("Example: python train.py ./data")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    try:
        # Validate data path
        validate_data_path(data_path)
        
        # Load and preprocess data
        img_data, labels = load_and_preprocess_data(data_path)
        
        # Create model
        model = create_model()
        
        # Train model
        history = train_model(model, img_data, labels)
        
        # Save model and history
        save_model(model, history)
        
        # Plot training history
        plot_training_history(history)
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
