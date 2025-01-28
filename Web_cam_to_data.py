import cv2
import mediapipe as mp
import csv
import time
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # Webcam

# Assume this CSV file already exists (with the proper header, if needed).
csv_filename = 'C:/Users/green/Downloads/AS-main/AS-main/Corrected_CSV_Data/T_asl_letter_data_corrected.csv'
count = 0
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    
    while True:
        success, frame = cap.read()
        if not success:
            continue

        # Flip horizontally so it looks more like a mirror
        frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Optional: draw landmarks on the frame so you can see them
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # Show the webcam feed
        cv2.imshow("Press R to record, Q to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # If 'R' is pressed, append hand data to CSV
        if key == ord('r'):
            if results.multi_hand_landmarks:
                # Take the first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]

                # Collect x,y,z for all 21 landmarks
                row_data = []
                for i in range(21):
                    lm = hand_landmarks.landmark[i]
                    row_data += [lm.x, lm.y, lm.z]

                # Append to the existing CSV (no new header)
                with open(csv_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row_data)
                count += 1
                print("Frame appended to CSV.",count)
                time.sleep(0.01)


        # If 'Q' is pressed, quit
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
