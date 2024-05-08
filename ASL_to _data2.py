import cv2
import mediapipe as mp
import os
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def extract_landmarks(image_path):
    # Read image
    image = cv2.imread(image_path)
    # Convert the image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image with MediaPipe
    results = hands.process(rgb_image)
    # Extract landmarks
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y, landmark.z))
    return landmarks

def save_landmarks_to_csv(image_folder, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['x', 'y', 'z'])  # Write header
        for filename in os.listdir(image_folder):
            image_path = os.path.join(image_folder, filename)
            landmarks = extract_landmarks(image_path)
            for landmark in landmarks:
                csv_writer.writerow(landmark)

# Example usage
image_folder = 'A\lit'
output_csv = 'A_asl_letter_data.csv'
save_landmarks_to_csv(image_folder, output_csv)