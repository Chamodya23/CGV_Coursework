# Hand Gesture RPS Game 👋🎮
_hand-gesture-rps_ is an interactive computer vision application that allows you to play Rock-Paper-Scissors against your computer using real hand gestures. Powered by MediaPipe and OpenCV, this game recognizes your hand gestures in real-time through your webcam, determines your move, and competes against a randomized computer opponent.

✨ Features

* Real-time hand gesture recognition using MediaPipe's hand tracking
* Visual processing pipeline with threshold adjustment and visualization options
* Classic RPS gameplay with intuitive controls
* Extended mode supporting Rock-Paper-Scissors-Lizard-Spock for added complexity
* Interactive countdown system with clear game state transitions
* Performance statistics tracking wins, losses, and ties
* Screenshot capability to capture memorable moments
* Customizable threshold settings for different lighting conditions

🔧 **Technologies**

* Python (3.8+)
* OpenCV (4.11.0+) for image processing and UI
* MediaPipe for hand landmark detection
* NumPy for numerical operations
* Random for computer move generation

📋 Requirements

* Webcam or camera device
* Python 3.8 or higher
* Libraries: OpenCV, MediaPipe, NumPy

🚀 Installation

1. Clone the Repository
git clone https://github.com/Chamodya23/CGV_Coursework.git

2. Create a Virtual Environment
python -m venv venv

3. Install Dependencies
   pip install -r requirements.txt

4. Prepare Image Assets
   * Create an _images_ folder in the project directory
   * Add images for game moves (rock.png, paper.png, scissors.png)
   * For extended mode, add lizard.png and spock.png

🎮 How to Play

1. Start the Game
* bashpython rps.py

2. Controls

* SPACE: Start a new round
* e: Toggle extended mode (Rock-Paper-Scissors-Lizard-Spock)
* t: Toggle threshold visualization
* +/-: Adjust threshold values
* s: Take a screenshot
* ESC: Quit the game

3. Gameplay

* Position your hand in front of the camera
* When you press SPACE, a 3-second countdown begins
* Form your gesture (rock, paper, scissors) before the countdown ends
* The system will detect your gesture and determine the winner
* Results are displayed for a few seconds before returning to the waiting state

🎲 Extended Mode
In extended mode, you can play Rock-Paper-Scissors-Lizard-Spock with these additional gestures:

* Spock: Extend your thumb and pinky (Star Trek salute)
* Lizard: Extend your thumb and index finger (like a puppet mouth)

Remember the rules:

* Scissors cuts Paper, Paper covers Rock
* Rock crushes Lizard, Lizard poisons Spock
* Spock smashes Scissors, Scissors decapitates Lizard
* Lizard eats Paper, Paper disproves Spock
* Spock vaporizes Rock, Rock crushes Scissors

