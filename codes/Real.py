# check.py

from fileinput import filename
import sys
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import logging
import os
import time
import io
from gtts import gTTS
import google.generativeai as genai
from dotenv import load_dotenv
# ... imports ...

# --- PATH CONFIGURATION (NEW) ---
# Get the folder where this script (Real.py) is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

# Define paths to your folders
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
ENV_PATH = os.path.join(PROJECT_ROOT, '.env')

# 1. Load the secret .env file from the root
load_dotenv(dotenv_path=ENV_PATH)

# ... rest of API_KEY setup ...

# 1. Load the secret .env file
load_dotenv()

# 2. Get the key safely from the environment
API_KEY = os.getenv("GEMINI_API_KEY")

# Check if key was found (Good for debugging)
if not API_KEY:
    print("Error: API Key not found. Make sure .env file exists.")
else:
    genai.configure(api_key=API_KEY)
    
# Initialize the model
generation_config = {
  "temperature": 1.0,  # <--- CHANGED TO 1.0
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 100,
}
gemini_model = genai.GenerativeModel("gemini-1.5-pro", generation_config=generation_config)

# ----------------------------- #
#       Configure Encoding
# ----------------------------- #

# Set standard output to UTF-8 to handle Unicode characters
try:
    # For Python 3.7 and above
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    # For Python versions below 3.7
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ----------------------------- #
#       Configure Logging
# ----------------------------- #

# Configure logging to write to a file with UTF-8 encoding
# Configure logging to write to the 'logs' folder
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, 'gesture_detection.log'),  # <--- UPDATED PATH
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ----------------------------- #
#       Load Pretrained Model
# ----------------------------- #

# Paths to the saved model and artifacts
# Paths to the saved model and artifacts (Updated to use MODELS_DIR)
MODEL_PATH = os.path.join(MODELS_DIR, 'SGT.h5')
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

# Load the trained model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info(f"Loaded model from '{MODEL_PATH}' successfully.")
except Exception as e:
    logging.error(f"Failed to load model from '{MODEL_PATH}': {e}")
    print(f"Error: Failed to load model from '{MODEL_PATH}'. Check logs for details.")
    sys.exit(1)

# Load the Label Encoder and Scaler
try:
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    logging.info(f"Loaded label encoder from '{LABEL_ENCODER_PATH}' and scaler from '{SCALER_PATH}' successfully.")
except Exception as e:
    logging.error(f"Failed to load label encoder or scaler: {e}")
    print("Error: Failed to load label encoder or scaler. Check logs for details.")
    sys.exit(1)

# ----------------------------- #
#       Initialize MediaPipe
# ----------------------------- #

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Hands
hands = mp_hands.Hands(
    static_image_mode=False,       # For real-time detection
    max_num_hands=1,               # Detect one hand
    min_detection_confidence=0.5,  # Minimum confidence for detection
    min_tracking_confidence=0.5    # Minimum confidence for tracking
)

# ----------------------------- #
#       Initialize Webcam
# ----------------------------- #

cap = cv2.VideoCapture(0)  # 0 is the default webcam

if not cap.isOpened():
    logging.error("Error: Could not open webcam.")
    print("Error: Could not open webcam.")
    sys.exit(1)

# ----------------------------- #
#       Initialize TTS Engine
# ----------------------------- #

# # Initialize the TTS engine
# tts_engine = pyttsx3.init()

# # Optionally, set properties like voice, rate, and volume
# tts_engine.setProperty('rate', 150)    # Speech rate
# tts_engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

# ----------------------------- #
#       Initialize State Variables
# ----------------------------- #

current_letter = None
word = ""
sentence = ""

# Debounce parameters
DEFAULT_DEBOUNCE_TIME = 0.5  # seconds for general gestures
DEBOUNCE_TIMES = {
    'confirm': 1.0,  # seconds cooldown for 'confirm' gesture
    'space': 1.0,    # seconds cooldown for 'space' gesture
    'speak': 1.5     # seconds cooldown for 'speak' gesture
}
last_gesture_time = {}  # Dictionary to store the last detection time for each gesture

# ----------------------------- #
#       Prediction and TTS Functions
# ----------------------------- #

def is_debounced(gesture_label, debounce_time=DEFAULT_DEBOUNCE_TIME):
    """
    Checks if the gesture can be registered based on the debounce time.
    
    Args:
        gesture_label (str): The label of the detected gesture.
        debounce_time (float): The debounce time in seconds.
    
    Returns:
        bool: True if the gesture can be registered, False otherwise.
    """
    current_time = time.time()
    if gesture_label in last_gesture_time:
        elapsed_time = current_time - last_gesture_time[gesture_label]
        if elapsed_time < debounce_time:
            return False
    last_gesture_time[gesture_label] = current_time
    return True

def correct_text_with_gemini(raw_text):
    """
    Sends raw text to Gemini with strict instructions to SPLIT words.
    """
    if not raw_text.strip():
        return ""
    
    try:
        prompt = (
                    "Task: Insert spaces into the text below to form a valid English sentence.\n"
                    "The input comes from a gesture glove that deletes all spaces.\n"
                    "You MUST split the string into separate words.\n\n"
                    "Examples:\n"
                    "Input: IAMFINE\nOutput: I am fine\n"
                    "Input: HELLOWORLD\nOutput: Hello World\n"
                    "Input: WHATISYOURNAME\nOutput: What is your name?\n"
                    "Input: GOODMORNING\nOutput: Good Morning\n\n"
                    f"Input: {raw_text}\n"
                    "Output:"
                )
        
        response = gemini_model.generate_content(prompt)
        corrected = response.text.strip()
        
        # Clean up "Output:" if present
        if corrected.lower().startswith("output:"):
            corrected = corrected[7:].strip()
            
        logging.info(f"Gemini Correction: {raw_text} -> {corrected}")
        return corrected
    except Exception as e:
        logging.error(f"Gemini API Error: {e}")
        return raw_text

def speak_sentence(sentence):
    """
    Converts text to speech using Google TTS and plays it on Mac.
    """
    if not sentence.strip():
        return

    try:
        # Create audio file from text
        tts = gTTS(text=sentence, lang='en', slow=False)
        filename = "voice_output.mp3"
        
        # Clean up old file
        if os.path.exists(filename):
            os.remove(filename)
            
        tts.save(filename)
        
        # Play audio (Mac version)
        print(f"Speaking: {sentence}")
        os.system(f"afplay '{filename}'")
        
        logging.info(f"Spoken Sentence: {sentence}")
    except Exception as e:
        logging.error(f"TTS Error: {e}")

def predict_gesture(keypoints):
    """
    Preprocesses keypoints and predicts the gesture label.

    Args:
        keypoints (list or np.array): Flattened list of keypoint coordinates.

    Returns:
        str: Predicted gesture label.
    """
    try:
        # Convert to numpy array and reshape for scaler
        keypoints = np.array(keypoints).reshape(1, -1)

        # Scale the keypoints using the loaded scaler
        keypoints_scaled = scaler.transform(keypoints)

        # Predict using the loaded model
        predictions = model.predict(keypoints_scaled)
        predicted_class = np.argmax(predictions, axis=1)

        # Decode the predicted class to label
        predicted_label = label_encoder.inverse_transform(predicted_class)[0]

        return predicted_label

    except Exception as e:
        logging.error(f"Error in predict_gesture: {e}")
        return "Prediction_Error"

# ----------------------------- #
#       Main Loop
# ----------------------------- #

logging.info("=== Real-Time Hand Gesture Recognition Started ===")
print("=== Real-Time Hand Gesture Recognition ===")
print("Press 'ESC' to quit. Press 'C' to confirm the letter.\n")
logging.info("Press 'ESC' to quit. Press 'C' to confirm the letter.")

while True:
    ret, frame = cap.read()
    if not ret:
        logging.warning("Failed to grab frame.")
        continue  # Skip to the next iteration

    # Flip the frame horizontally for a mirror view
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Extract keypoints
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])

            # Predict the gesture
            predicted_gesture = predict_gesture(keypoints)

            # Display detected letter
            if predicted_gesture.upper() in [chr(i) for i in range(65, 91)]:  # A-Z
                current_letter = predicted_gesture.upper()
                cv2.putText(
                    frame,
                    f"Detected Letter: {current_letter}",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA
                )

    # Display the current word and sentence
    cv2.putText(
        frame,
        f"Word: {word}",
        (10, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )
    # cv2.putText(
    #     frame,
    #     f"Sentence: {sentence}",
    #     (10, 190),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.8,
    #     (0, 255, 255),
    #     2,
    #     cv2.LINE_AA
    # )

    # Display instructions on the frame
    cv2.putText(
        frame,
        "Press 'ESC' to quit. Press 'C' to confirm the letter.",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Show the frame
    cv2.imshow('Real-Time Hand Gesture Recognition', frame)

# ----------------------------- #
#       Key press handling
# ----------------------------- #

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key to exit
        logging.info("ESC pressed. Exiting program.")
        print("Exiting program.")
        break
    elif key == ord('c') or key == ord('C'):
        if current_letter:
            word += current_letter
            logging.info(f"Added Letter: {word}")
            current_letter = None
    elif key == ord('t') or key == ord('T'):
        word += " "  # Add a space
        logging.info("Space added.")

    elif key == ord('s') or key == ord('S'):
            if word:
                print(f"\nRaw Input: {word}")
                print("Gemini is thinking...") 
                
                # 1. Send to Gemini
                corrected_sentence = correct_text_with_gemini(word)
                print(f"Corrected: {corrected_sentence}")
                
                # 2. Update the display variable
                word = corrected_sentence 
                
                # 3. Speak the result
                speak_sentence(word)
                
                # Optional: Clear word after speaking
                word = ""



# ----------------------------- #
#       Release Resources
# ----------------------------- #

cap.release()
cv2.destroyAllWindows()
hands.close()
# tts_engine.stop()
logging.info("=== Real-Time Hand Gesture Recognition Ended ===")