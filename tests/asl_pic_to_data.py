import cv2
import mediapipe as mp
import os
import csv

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Directory containing your images
image_dir = 'A\lit'

# Output CSV file
csv_filename = 'asl_letters.csv'

# Open CSV file for writing
with open(csv_filename, 'w') as csvfile:
    # Create CSV writer object
    csv_writer = csv.writer(csvfile)
    
    # Write header row
    csv_writer.writerow(['Image', 'Letter', 'Landmarks'])
    
    # Loop through each image file
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Read image
            img_path = os.path.join(image_dir, filename)
            img = cv2.imread(img_path)
            
            # Convert image to RGB (MediaPipe Hands model requires RGB input)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect hand landmarks
            results = hands.process(img_rgb)
            
            # If hand landmarks are detected
            if results.multi_hand_landmarks:
                # Get landmarks for the first hand (assuming only one hand in the image)
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Convert landmarks to a single string
                landmarks_str = ','.join([f'{lm.x},{lm.y},{lm.z}' for lm in hand_landmarks.landmark])
                
                # Write image filename, letter, and landmarks to CSV
                csv_writer.writerow([filename, 'A', landmarks_str])
                
print("Hand landmarks data saved to", csv_filename)