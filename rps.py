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

 def create_thresholded_hand(self, frame, hand_landmarks):
        """Create a thresholded image of the hand on a black background"""
        if hand_landmarks is None:
            return np.zeros_like(frame)
        
        # Extract hand region
        hand_image = self.extract_hand_region(frame, hand_landmarks)
        
        # Convert to grayscale
        gray = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)
        
        # Apply binary threshold
        _, thresh = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)
        
        # Convert back to BGR for display
        thresh_colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        # Draw landmarks on the thresholded image for better visualization
        thresh_with_landmarks = thresh_colored.copy()
        self.mp_drawing.draw_landmarks(
            thresh_with_landmarks,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
        
        return thresh_with_landmarks

def create_processing_visualization(self, frame):
        h, w = frame.shape[:2]
        # Create a 2x2 grid for visualization
        grid = np.zeros((h, w*2, 3), dtype=np.uint8)

        # Original frame in top-left (without gesture label, which will be added separately)
        grid[:h//2, :w] = frame[:h//2]

        # Thresholded hand in top-right if hand is detected
        if self.hand_landmarks and self.show_threshold:
            thresh_hand = self.create_thresholded_hand(frame, self.hand_landmarks)
            grid[:h//2, w:w*2] = cv2.resize(thresh_hand, (w, h//2))
        else:
            # Convert to grayscale for processing visualization
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            grid[:h//2, w:w*2] = cv2.resize(gray_bgr, (w, h//2))

        # Hand landmarks visualization in bottom-left (only draw once)
        hand_vis = frame.copy()
        if self.hand_landmarks:
            self.mp_drawing.draw_landmarks(
                hand_vis, 
                self.hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS)
        grid[h//2:, :w] = hand_vis[h//2:]

        # Game state in bottom-right
        info_display = np.zeros((h//2, w, 3), dtype=np.uint8)

        # Show countdown or result
        if self.state == GameState.COUNTDOWN:
            time_left = int(self.countdown_end - time.time()) + 1
            if time_left > 0:
                cv2.putText(info_display, str(time_left), (w//2-50, h//4+20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)
        elif self.state == GameState.RESULT:
            # Show player's move
            player_move_text = f"Your move: {self.player_move}"
            cv2.putText(info_display, player_move_text, (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show computer's move
            computer_move_text = f"Computer: {self.computer_move}"
            cv2.putText(info_display, computer_move_text, (20, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show game result
            result_color = (0, 255, 0) if self.result == "You win!" else \
                        (0, 0, 255) if self.result == "Computer wins!" else (255, 255, 255)
            cv2.putText(info_display, self.result, (20, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, result_color, 2)
            
            # Show score
            score_text = f"Score: You {self.player_score} - {self.computer_score} Computer"
            cv2.putText(info_display, score_text, (20, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show mode
        mode_text = "Mode: " + ("Standard" if not self.extended_mode else "Extended (Rock-Paper-Scissors-Lizard-Spock)")
        cv2.putText(info_display, mode_text, (20, h//4-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add the status display to bottom-right
        grid[h//2:, w:w*2] = info_display
        
        return grid
    
     def detect_gesture(self, hand_landmarks):
            """Detect the gesture based on MediaPipe hand landmarks"""
        # Get tip and pip landmarks for each finger
        finger_tips = [
            hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        ]
        
        finger_pips = [
            hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]
        ]
        
        # Check if fingers are extended
        fingers_extended = []
        for tip, pip in zip(finger_tips, finger_pips):
            if tip.y < pip.y:  # If tip is above pip (y increases downward)
                fingers_extended.append(True)
            else:
                fingers_extended.append(False)
        
        # Special case for thumb - check if thumb tip is to the right/left of the thumb ip
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        thumb_tip = finger_tips[0]
        thumb_ip = finger_pips[0]
        
        # Determine if the hand is left or right
        if wrist.x < thumb_tip.x:  # Right hand
            fingers_extended[0] = thumb_tip.x > thumb_ip.x
        else:  # Left hand
            fingers_extended[0] = thumb_tip.x < thumb_ip.x
        
        # Detect basic gestures
        if not any(fingers_extended):
            return "rock"
        elif all(fingers_extended):
            return "paper"
        elif fingers_extended[1] and fingers_extended[2] and not fingers_extended[0] and not fingers_extended[3] and not fingers_extended[4]:
            return "scissors"
        elif self.extended_mode:
            # For extended mode (Rock, Paper, Scissors, Lizard, Spock)
            if fingers_extended[0] and fingers_extended[4] and not fingers_extended[1] and not fingers_extended[2] and not fingers_extended[3]:
                return "spock"
            elif fingers_extended[0] and fingers_extended[1] and not fingers_extended[2] and not fingers_extended[3] and not fingers_extended[4]:
                return "lizard"
        
        return None
    
    def determine_winner(self, user_choice, computer_choice):
        """Determine the winner based on Rock-Paper-Scissors(-Lizard-Spock) rules"""
        if user_choice == computer_choice:
            return "Tie!"
        
        # Rules for standard Rock-Paper-Scissors
        if not self.extended_mode:
            if (user_choice == "rock" and computer_choice == "scissors") or \
               (user_choice == "paper" and computer_choice == "rock") or \
               (user_choice == "scissors" and computer_choice == "paper"):
                return "You win!"
            else:
                return "Computer wins!"
        else:
            # Rules for Rock-Paper-Scissors-Lizard-Spock
            # Scissors cuts Paper, Paper covers Rock, Rock crushes Lizard, 
            # Lizard poisons Spock, Spock smashes Scissors, Scissors decapitates Lizard, 
            # Lizard eats Paper, Paper disproves Spock, Spock vaporizes Rock, Rock crushes Scissors
            win_conditions = {
                "scissors": ["paper", "lizard"],
                "paper": ["rock", "spock"],
                "rock": ["lizard", "scissors"],
                "lizard": ["spock", "paper"],
                "spock": ["scissors", "rock"]
            }
            
            if computer_choice in win_conditions.get(user_choice, []):
                return "You win!"
            else:
                return "Computer wins!"
            
            def overlay_computer_choice(self, display, computer_move):
        """Overlay computer's choice on the game display"""
        
        # Make sure computer_move is valid
        if not computer_move or computer_move not in self.extended_choices:
            print(f"Invalid computer move: {computer_move}")
            return display
        
        # Get the appropriate image based on computer's move
        try:
            if computer_move == "rock":
                img_path = self.rock_img_path
            elif computer_move == "paper":
                img_path = self.paper_img_path
            elif computer_move == "scissors":
                img_path = self.scissors_img_path
            elif computer_move in self.gesture_images:
                # Use the preloaded images for extended mode gestures
                img = self.gesture_images[computer_move]
                if img is not None:
                    # Skip to the resizing part if we have a valid preloaded image
                    target_width = int(display.shape[1] * 0.3)
                    ratio = img.shape[1] / img.shape[0]
                    target_height = int(target_width / ratio)
                    img_resized = cv2.resize(img, (target_width, target_height))
                    
                    # Continue with the overlay logic below
                    img_channels = len(img_resized.shape)
                    has_alpha = False
                    
                    if img_channels == 3:
                        # RGB image, no alpha channel
                        pass
                    elif img_channels > 3 and img_resized.shape[2] == 4:
                        # Image with alpha channel
                        has_alpha = True
                    else:
                        # Grayscale image or other format
                        # Convert to RGB to ensure compatibility
                        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
                    
                    # Position image in the top-right corner with some padding
                    x_offset = display.shape[1] - target_width - 20
                    y_offset = 20
                    
                    # Ensure we don't go out of bounds
                    if y_offset + target_height > display.shape[0] or x_offset + target_width > display.shape[1]:
                        target_height = min(target_height, display.shape[0] - y_offset)
                        target_width = min(target_width, display.shape[1] - x_offset)
                        img_resized = cv2.resize(img, (target_width, target_height))
                    
                    # Create a region of interest
                    roi = display[y_offset:y_offset+target_height, x_offset:x_offset+target_width]
                    
                    # Overlay the image properly handling alpha if present
                    if has_alpha:
                        # Extract alpha channel
                        alpha = img_resized[:, :, 3] / 255.0
                        
                        # Extract RGB channels
                        for c in range(3):
                            roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * img_resized[:, :, c]
                    else:
                        # Simple overlay for images without alpha
                        roi[:] = img_resized[:, :, :3]
                    
                    return display
            
            # Check if image path exists for basic gestures
            if not os.path.exists(img_path):
                print(f"Error: Image file not found at {img_path}")
                # Draw text instead of image as fallback
                cv2.putText(display, f"Computer: {computer_move.upper()}", 
                        (display.shape[1] - 200, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                return display
                
            # Load and resize the image
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Error: Could not load image at {img_path}")
                # Draw text instead of image as fallback
                cv2.putText(display, f"Computer: {computer_move.upper()}", 
                        (display.shape[1] - 200, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                return display
                
            # Calculate target size (e.g., 30% of the display width)
            target_width = int(display.shape[1] * 0.3)
            ratio = img.shape[1] / img.shape[0]
            target_height = int(target_width / ratio)
            
            img_resized = cv2.resize(img, (target_width, target_height))
            
            # Check image dimensions and handle various image formats
            img_channels = len(img_resized.shape)
            has_alpha = False
            
            if img_channels == 3:
                # RGB image, no alpha channel
                pass
            elif img_channels > 3 and img_resized.shape[2] == 4:
                # Image with alpha channel
                has_alpha = True
            else:
                # Grayscale image or other format
                # Convert to RGB to ensure compatibility
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
            
            # Position image in the top-right corner with some padding
            x_offset = display.shape[1] - target_width - 20
            y_offset = 20
            
            # Ensure we don't go out of bounds
            if y_offset + target_height > display.shape[0] or x_offset + target_width > display.shape[1]:
                target_height = min(target_height, display.shape[0] - y_offset)
                target_width = min(target_width, display.shape[1] - x_offset)
                img_resized = cv2.resize(img_resized, (target_width, target_height))
            
            # Create a region of interest
            roi = display[y_offset:y_offset+target_height, x_offset:x_offset+target_width]
            
            # Overlay the image properly handling alpha if present
            if has_alpha:
                # Extract alpha channel
                alpha = img_resized[:, :, 3] / 255.0
                
                # Extract RGB channels
                for c in range(3):
                    roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * img_resized[:, :, c]
            else:
                # Simple overlay for images without alpha
                roi[:] = img_resized[:, :, :3]
                
            return display
        except Exception as e:
            print(f"Error overlaying computer choice: {e}")
            # Draw text instead of image as fallback
            cv2.putText(display, f"Computer: {computer_move.upper()}", 
                    (display.shape[1] - 200, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            return display  # Return original display on error