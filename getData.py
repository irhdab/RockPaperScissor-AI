"""
Data Collection Script for Rock Paper Scissors AI
Collects training images for gesture recognition model

This script captures images from webcam for training the neural network.
It creates directories for each gesture type and saves images with proper naming.
"""

import numpy as np
import cv2
import os
import sys

def create_data_directory(label):
    """
    Create directory for storing data with proper path handling
    
    Args:
        label (str): The gesture label (rock, paper, scissor)
        
    Returns:
        str: Path to the created directory
        
    Raises:
        SystemExit: If directory creation fails
    """
    current_path = os.getcwd()
    save_path = os.path.join(current_path, label)
    
    try:
        # Create directory if it doesn't exist, ignore if it does
        os.makedirs(save_path, exist_ok=True)
        print(f"Data directory created/verified: {save_path}")
        return save_path
    except Exception as e:
        print(f"Error creating directory: {e}")
        sys.exit(1)

def validate_arguments():
    """
    Validate command line arguments
    
    Returns:
        tuple: (start_index, end_index) - validated index range
        
    Raises:
        SystemExit: If arguments are invalid
    """
    if len(sys.argv) != 4:
        print("Usage: python getData.py <label> <startIndex> <endIndex>")
        print("Example: python getData.py rock 1 100")
        sys.exit(1)
    
    try:
        start_index = int(sys.argv[2])
        end_index = int(sys.argv[3])
        
        # Validate index range
        if start_index < 0 or end_index < start_index:
            raise ValueError("Invalid index range")
            
        return start_index, end_index
    except ValueError as e:
        print(f"Error with index arguments: {e}")
        sys.exit(1)

def capture_images(save_path, label, start_index, end_index):
    """
    Capture images from webcam and save them to disk
    
    Args:
        save_path (str): Directory to save images
        label (str): Gesture label for naming
        start_index (int): Starting image number
        end_index (int): Ending image number
        
    Returns:
        int: Number of images actually captured
    """
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)
    
    print(f"Hit Space to Capture Image for {label}")
    print(f"Will capture {end_index - start_index + 1} images")
    
    current_count = start_index
    captured_count = 0
    
    try:
        while current_count <= end_index:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
                
            # Display the capture area (Region of Interest)
            # This shows the user exactly what area will be captured
            display_frame = frame[50:350, 150:450].copy()
            cv2.imshow(f'Get Data: {label}', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                # Save the image with proper path handling
                image_filename = f"{label}{current_count}.jpg"
                image_path = os.path.join(save_path, image_filename)
                
                try:
                    # Save the cropped region of interest
                    cv2.imwrite(image_path, frame[50:350, 150:450])
                    print(f"Captured: {image_path}")
                    current_count += 1
                    captured_count += 1
                except Exception as e:
                    print(f"Error saving image: {e}")
            elif key == ord('q'):
                print("Data collection cancelled by user")
                break
                
    except KeyboardInterrupt:
        print("\nData collection interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Data collection completed. {captured_count} images saved.")
        return captured_count

def main():
    """
    Main function to orchestrate data collection process
    
    This function:
    1. Validates command line arguments
    2. Creates necessary directories
    3. Captures images from webcam
    4. Handles errors gracefully
    """
    # Validate arguments
    start_index, end_index = validate_arguments()
    label = sys.argv[1]
    
    # Validate label
    valid_labels = ['rock', 'paper', 'scissor']
    if label not in valid_labels:
        print(f"Error: Label must be one of {valid_labels}")
        sys.exit(1)
    
    # Create directory
    save_path = create_data_directory(label)
    
    # Capture images
    captured_count = capture_images(save_path, label, start_index, end_index)
    
    if captured_count == 0:
        print("Warning: No images were captured")
    else:
        print(f"Successfully captured {captured_count} images for {label}")

if __name__ == "__main__":
    main()