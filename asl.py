import cv2
import mediapipe as mp
import math
import time
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

mp_drawig = mp.solutions.drawing_utils

mp_hands = mp.solutions.hands
hand = mp_hands.Hands()
start = time.time()
end = 1 
while True:
    success, frame = cap.read()
    if success:
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hand.process(RGB_frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmark
                mp_drawig.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


                #FPS function

                end = time.time()  # Set end time after processing each frame
                fps = math.ceil(1 / (end - start))
                start = time.time() 

                # Get landmarks for index finger and base of the hand
                #index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                #wrist_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                # Calculate distance between index finger and base of the hand in pixels
                #distance = math.sqrt((index_finger_landmark.x - wrist_landmark.x) ** 2 + (index_finger_landmark.y - wrist_landmark.y) ** 2)
                
                # Draw a line between index finger tip and wrist
                #finger_pos = (int(index_finger_landmark.x * frame.shape[1]), int(index_finger_landmark.y * frame.shape[0]))
                #wrist_pos = (int(wrist_landmark.x * frame.shape[1]), int(wrist_landmark.y * frame.shape[0]))
                #cv2.line(frame, finger_pos, wrist_pos, (0, 255, 0), 2)
                
                #ASL function
                #asl_gesture = asl_recognize_gesture(hand_region)



                # Display distance on the frame
                cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                #cv2.putText(frame, f'Distance: {distance:.2f} pixels', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.imshow("capture image", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()

