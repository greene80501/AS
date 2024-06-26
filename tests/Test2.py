import cv2
import mediapipe as mp
import csv
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5)  # Adjust confidence as needed

def extract_landmarks(image):
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
    # Ensure we have exactly 21 landmarks
    if len(landmarks) != 21:
        return None
    return landmarks

def compare_landmarks(landmarks, csv_file):
    with open(csv_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip header row
        min_distance = float('inf')
        closest_letter = ''
        for row in csv_reader:
            csv_landmarks = [float(coord) for coord in row[1:]]  # Assuming the first element is the letter
            # Ensure both sources have exactly 21 landmarks
            if len(csv_landmarks) != 21:
                continue
            distance = np.linalg.norm(np.array(landmarks) - np.array(csv_landmarks))
            if distance < min_distance:
                min_distance = distance
                closest_letter = row[0]  # Assuming the first element is the letter
        return closest_letter

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Extract landmarks
    landmarks = extract_landmarks(frame)

    if landmarks:
        # Draw a rectangle around the hand
        # Calculate bounding box coordinates
        min_x = int(min(landmark[0] for landmark in landmarks) * frame.shape[1])
        min_y = int(min(landmark[1] for landmark in landmarks) * frame.shape[0])
        max_x = int(max(landmark[0] for landmark in landmarks) * frame.shape[1])
        max_y = int(max(landmark[1] for landmark in landmarks) * frame.shape[0])
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 0, 0), 2)

        # Compare landmarks with CSV
        closest_letter = compare_landmarks([landmark[:2] for landmark in landmarks], 'A_asl_letter_data.csv')

        # Display the closest letter above the hand if it matches 'A'
        if closest_letter == 'A':
            print("A")
            #cv2.putText(frame, closest_letter, (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('ASL Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()