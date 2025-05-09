import cv2
import numpy as np
import mediapipe as mp
import random
import time
import os
from datetime import datetime

from enum import Enum

# Define game states as an enum
class GameState(Enum):
    WAITING = 0     # Waiting for player to start game
    COUNTDOWN = 1   # Countdown before capturing gesture
    CAPTURE = 2     # Capturing the player's gesture
    RESULT = 3      # Displaying the result

class RPSGame:
    def __init__(self, resolution=(640,480), camera_index=0):
        # Your existing initialization code
        self.capture = cv2.VideoCapture(camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        
        # Game state variables
        self.state = "waiting"  # Possible states: waiting, countdown, playing, result
        self.countdown_time = 0
        self.last_countdown_update = 0
        self.player_move = None
        self.computer_move = None
        self.result = None
        self.score = {"player": 0, "computer": 0, "ties": 0}
        self.game_history = []
        
        # Initialize image paths - Add these lines
        self.rock_img_path = "images/rock.png"  # Update with your actual path
        self.paper_img_path = "images/paper.png"  # Update with your actual path
        self.scissors_img_path = "images/scissors.png"  # Update with your actual path
        
        # Check if image files exist
        for img_path in [self.rock_img_path, self.paper_img_path, self.scissors_img_path]:
            if not os.path.exists(img_path):
                print(f"Warning: Image file not found: {img_path}")
            

        # Initialize game state
        self.state = GameState.WAITING
        self.player_move = None
        self.computer_move = None
        self.result = None
        self.countdown_end = 0
        self.result_display_end = 0
        self.player_score = 0
        self.computer_score = 0
        self.extended_mode = False
        self.last_screenshot_time = 0
    
        # Initialize hand landmarks attribute
        self.hand_landmarks = None
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Game variables
        self.choices = ["rock", "paper", "scissors"]
        self.extended_choices = ["rock", "paper", "scissors", "lizard", "spock"]
        self.user_choice = None
        self.computer_choice = None
        self.result = None
        self.game_active = False
        self.countdown_started = False
        self.countdown_time = 0
        self.result_time = 0
        self.show_result = False
        self.extended_mode = False  # Set to True to play Rock, Paper, Scissors, Lizard, Spock
        
        # Thresholding parameters
        self.show_threshold = True
        self.threshold_value = 150
        
        # Load gesture images
        self.gesture_images = {}
        for gesture in self.extended_choices:
            path = f"images/{gesture}.png"
            if os.path.exists(path):
                self.gesture_images[gesture] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            else:
                print(f"Warning: Could not find image for {gesture} at {path}")
        
        # Create directory for images if it doesn't exist
        if not os.path.exists('images'):
            os.makedirs('images')
            print("Created 'images' directory. Please add gesture images there.")
        
        # Win/loss statistics
        self.stats = {"wins": 0, "losses": 0, "ties": 0, "total": 0}
        
        # Processing display flags
        self.show_binary = False
        self.show_grayscale = False
        self.show_edges = False
        self.show_contours = False

 def extract_hand_region(self, frame, hand_landmarks):
        """Extract the hand region from the frame using landmarks"""
        if hand_landmarks is None:
            return np.zeros_like(frame)
            
        h, w = frame.shape[:2]
        
        # Get hand bounding box
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)
        
        # Add padding to the bounding box
        padding = 30
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Create a mask for the hand region
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create convex hull of landmarks for the hand mask
        points = []
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            points.append([x, y])
            
        if len(points) > 0:
            points_array = np.array(points)
            convex_hull = cv2.convexHull(points_array)
            cv2.drawContours(mask, [convex_hull], -1, 255, -1)
            
            # Apply morphological operations to smooth the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            # Create a black background image
            hand_image = np.zeros_like(frame)
            
            # Copy only the hand portion from the original frame
            for c in range(3):  # For each color channel
                hand_image[:, :, c] = cv2.bitwise_and(frame[:, :, c], mask)
                
            return hand_image
            
        return np.zeros_like(frame)