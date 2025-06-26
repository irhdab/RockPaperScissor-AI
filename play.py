"""
Rock Paper Scissors AI Game
Improved version with better error handling and user experience
"""

import json
import numpy as np
from skimage import io
import cv2
import random
import os
import sys
from tensorflow.keras.models import model_from_json

class RockPaperScissorsGame:
    def __init__(self, model_json_path="model.json", model_weights_path="model.h5"):
        """Initialize the game with model loading"""
        self.model = None
        self.model_json_path = model_json_path
        self.model_weights_path = model_weights_path
        self.cap = None
        
        # Game state
        self.rounds = 0
        self.bot_score = 0
        self.player_score = 0
        self.num_rounds = 3
        
        # Labels mapping
        self.shape_to_label = {
            'rock': np.array([1., 0., 0.]),
            'paper': np.array([0., 1., 0.]),
            'scissor': np.array([0., 0., 1.])
        }
        self.arr_to_shape = {np.argmax(self.shape_to_label[x]): x for x in self.shape_to_label.keys()}
        
        # Game options
        self.options = ['rock', 'paper', 'scissor']
        self.win_rule = {'rock': 'scissor', 'scissor': 'paper', 'paper': 'rock'}
        
        # Load model
        self.load_model()
        
    def load_model(self):
        """Load the trained model"""
        try:
            # Check if model files exist
            if not os.path.exists(self.model_json_path):
                raise FileNotFoundError(f"Model architecture file not found: {self.model_json_path}")
            
            if not os.path.exists(self.model_weights_path):
                raise FileNotFoundError(f"Model weights file not found: {self.model_weights_path}")
            
            # Load model architecture
            with open(self.model_json_path, 'r') as f:
                loaded_model_json = f.read()
            
            self.model = model_from_json(loaded_model_json)
            self.model.load_weights(self.model_weights_path)
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure model.json and model.h5 files are in the current directory")
            sys.exit(1)
    
    def prep_img(self, img):
        """Prepare image for prediction"""
        return cv2.resize(img, (300, 300)).reshape(1, 300, 300, 3)
    
    def update_score(self, player_play, bot_play, player_score, bot_score):
        """Update scores based on game rules"""
        if player_play == bot_play:
            return player_score, bot_score
        elif bot_play == self.win_rule[player_play]:
            return player_score + 1, bot_score
        else:
            return player_score, bot_score + 1
    
    def initialize_camera(self):
        """Initialize camera with error handling"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            sys.exit(1)
        
        # Warm up camera
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read from camera")
            sys.exit(1)
        
        # Test prediction
        try:
            self.model.predict(self.prep_img(frame[50:350, 100:400]))
            print("Camera initialized successfully!")
        except Exception as e:
            print(f"Error testing model prediction: {e}")
            sys.exit(1)
    
    def show_welcome_screen(self):
        """Show welcome screen and wait for user to start"""
        print("Welcome to Rock Paper Scissors AI!")
        print("Press SPACE to start the game")
        print("Press Q to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read from camera")
                break
            
            # Add welcome text
            frame = cv2.putText(frame, "Press SPACE to start", (160, 200), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA)
            frame = cv2.putText(frame, "Press Q to quit", (180, 250), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Rock Paper Scissors AI', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            elif key == ord('q'):
                return False
        
        return True
    
    def play_round(self, round_num):
        """Play a single round"""
        print(f"\nRound {round_num + 1} of {self.num_rounds}")
        
        pred = ""
        bot_play = ""
        
        # 90 frames = 3 seconds at 30fps
        for i in range(90):
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read from camera")
                return False
            
            # Countdown phase (first 3 seconds)
            if i // 30 < 3:
                countdown = 3 - (i // 30)
                frame = cv2.putText(frame, str(countdown), (320, 100), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 3, (250, 250, 0), 2, cv2.LINE_AA)
            
            # Prediction phase (3-3.5 seconds)
            elif i / 30 < 3.5:
                try:
                    prediction = self.model.predict(self.prep_img(frame[50:350, 100:400]))
                    pred = self.arr_to_shape[np.argmax(prediction)]
                except Exception as e:
                    print(f"Error making prediction: {e}")
                    pred = "unknown"
            
            # Bot decision phase (3.5 seconds)
            elif i / 30 == 3.5:
                bot_play = random.choice(self.options)
                print(f"Player: {pred} | Bot: {bot_play}")
            
            # Score update phase (4 seconds)
            elif i // 30 == 4:
                if pred in self.options:
                    self.player_score, self.bot_score = self.update_score(
                        pred, bot_play, self.player_score, self.bot_score
                    )
                break
            
            # Draw game elements
            cv2.rectangle(frame, (100, 150), (300, 350), (255, 255, 255), 2)
            frame = cv2.putText(frame, f"Player: {self.player_score} | Bot: {self.bot_score}", 
                              (120, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA)
            frame = cv2.putText(frame, f"Player: {pred}", (150, 140), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA)
            frame = cv2.putText(frame, f"Bot: {bot_play}", (300, 140), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Rock Paper Scissors AI', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False
        
        return True
    
    def show_final_results(self):
        """Show final game results"""
        if self.player_score > self.bot_score:
            winner = "You Won!!"
        elif self.player_score == self.bot_score:
            winner = "It's a Tie!"
        else:
            winner = "Bot Won!"
        
        print(f"\nFinal Score - Player: {self.player_score} | Bot: {self.bot_score}")
        print(f"Result: {winner}")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Add result text
            frame = cv2.putText(frame, winner, (230, 150), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA)
            frame = cv2.putText(frame, "Press Q to quit", (190, 200), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA)
            frame = cv2.putText(frame, f"Player: {self.player_score} | Bot: {self.bot_score}", 
                              (120, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Rock Paper Scissors AI', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def run(self):
        """Main game loop"""
        try:
            # Initialize camera
            self.initialize_camera()
            
            # Show welcome screen
            if not self.show_welcome_screen():
                return
            
            # Play rounds
            for round_num in range(self.num_rounds):
                if not self.play_round(round_num):
                    break
            
            # Show final results
            self.show_final_results()
            
        except KeyboardInterrupt:
            print("\nGame interrupted by user")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            self.cleanup()

def main():
    """Main function"""
    print("Starting Rock Paper Scissors AI Game...")
    
    # Check for model files
    if not os.path.exists("model.json") or not os.path.exists("model.h5"):
        print("Error: Model files not found!")
        print("Please ensure model.json and model.h5 are in the current directory")
        print("Run train.py first to create these files")
        sys.exit(1)
    
    # Create and run game
    game = RockPaperScissorsGame()
    game.run()

if __name__ == "__main__":
    main()
