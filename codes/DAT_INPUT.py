import cv2
import mediapipe as mp
import os
import time
import csv

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Define labels: Letters A-Z and custom labels
letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
additional_labels = {
    '1': 'Confirm',
    '2': 'Space',
    '3': 'Speak'
}
all_labels = letters + list(additional_labels.values())

# Define key-to-label mapping
key_to_label = {ord(k): v for k, v in additional_labels.items()}
for letter in letters:
    key_to_label[ord(letter.lower())] = letter  # Lowercase keys
    key_to_label[ord(letter.upper())] = letter  # Uppercase keys

# ... imports ...

# --- PATH CONFIGURATION ---
# Get the folder where this script is (i.e., 'codes/')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

# Define the dataset path in the Project Root
dataset_path = os.path.join(PROJECT_ROOT, 'media_pipe_keypoints_dataset')

# Create directories for each label if they don't exist
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
    for label in all_labels:
        label_dir = os.path.join(dataset_path, label)
        os.makedirs(label_dir, exist_ok=True)
        print(f"Created directory: {label_dir}")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\n=== Keypoint Capture Instructions ===")
print("Press a key (A-Z or 1-3) to start capturing keypoints.")
print("Press 'ESC' to stop capturing and exit.\n")

# Initialize debounce variables
last_capture_time = 0
capture_delay = 0.3  # seconds
capturing_label = None  # Tracks the active capturing label

def save_keypoints_to_csv(label, keypoints):
    """Save keypoints to a CSV file with a unique filename using a timestamp."""
    label_path = os.path.join(dataset_path, label)
    timestamp = int(time.time() * 1000)  # Milliseconds timestamp
    csv_filename = os.path.join(label_path, f"{label}_{timestamp}.csv")

    # Save keypoints to CSV
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Landmark', 'X', 'Y', 'Z', 'Visibility'])  # CSV headers
        for idx, landmark in enumerate(keypoints.landmark):
            writer.writerow([idx, landmark.x, landmark.y, landmark.z, landmark.visibility])

    print(f"Saved keypoints: {csv_filename}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert BGR image to RGB for MediaPipe processing
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect hand landmarks
    result = hands.process(image_rgb)

    # Draw hand landmarks if detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # Display instructions on the frame
    cv2.putText(
        frame,
        "Press A-Z or 1-3 to start capturing. ESC to stop.",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Display capturing status if a label is active
    if capturing_label:
        cv2.putText(
            frame,
            f"Capturing: {capturing_label}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    # Show the frame
    cv2.imshow('MediaPipe Hand Detection', frame)

    # Wait for a keypress for 1ms
    key = cv2.waitKey(1) & 0xFF
    current_time = time.time()

    # Check if a key is pressed to start capturing
    if key in key_to_label:
        capturing_label = key_to_label[key]  # Set the capturing label
        print(f"Started capturing: {capturing_label}")

    # Capture data continuously if a label is active
    if capturing_label and result.multi_hand_landmarks:
        if (current_time - last_capture_time) > capture_delay:
            # Save the detected hand keypoints for the current frame
            for hand_landmarks in result.multi_hand_landmarks:
                save_keypoints_to_csv(capturing_label, hand_landmarks)

            last_capture_time = current_time

    # Stop capturing and exit if ESC is pressed
    if key == 27:
        if capturing_label:
            print(f"Stopped capturing: {capturing_label}")
            capturing_label = None
        else:
            print("Exiting program.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
