import cv2
import mediapipe as mp
import math
import time
import csv


csv_filename = 'asl_letters.csv'
# Function to compare landmarks
def compare_landmarks(hand_landmarks, csv_data):
    hand_landmarks_list = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
    for row in csv_data:
        csv_landmarks = [(float(row[i]), float(row[i+1]), float(row[i+2])) for i in range(2, len(row), 3)]
        if hand_landmarks_list == csv_landmarks:
            return True
    return False

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Open CSV file and read landmark data
    with open(csv_filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip header row
        csv_data = list(csv_reader)

    start = time.time()  # Set start time before the loop
    end = 1

    while True:
        success, frame = cap.read()
        if success:
            RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(RGB_frame)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
                    
                    # Compare landmarks with CSV data
                    if compare_landmarks(hand_landmarks.landmark, csv_data):
                        # If landmarks match, display "A" in the middle of the camera frame
                        cv2.putText(frame, 'A', (int(frame.shape[1]/2), int(frame.shape[0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            # Calculate FPS
            end = time.time()
            fps = math.ceil(1 / (end - start))
            start = time.time()

            # Display FPS
            cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Hand Tracking", frame)
            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
