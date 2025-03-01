import cv2
import mediapipe as mp
import math
import time
import tensorflow as tf
import os
import numpy as np
import predict

tf.keras.utils.disable_interactive_logging()

# Provide the path to the directory where the model is saved
model_path = os.curdir + "/asl_model_Test.keras"

# Load the saved model
loaded_model = tf.keras.models.load_model(model_path)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

start = time.time()  # Set start
end = 1
predicted_letter = None

frames = []

while True:
    success, frame = cap.read()
    frames.append(frame)
    if success and len(frames) == 24:
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hand.process(RGB_frame)
        
        if result.multi_hand_landmarks:
            p_this = list()

            for frame in frames:
                for hand_landmarks in result.multi_hand_landmarks:
                    for id, lm in enumerate(hand_landmarks.landmark):
                        p_this.append(float(lm.x))
                        p_this.append(float(lm.y))
                        p_this.append(float(lm.z))

            if len(p_this) == 1512:
                p_this = np.array(p_this).reshape(1, 24*63)
                predicted_letter = predict._predict(p_this, loaded_model)

                # Draw landmarks with black lines and white dots
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2))

                # Get bounding box coordinates
                bbox = cv2.boundingRect(np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]] for lm in hand_landmarks.landmark]).astype(int))


                # Draw black rectangle around the hand
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 0), 2)

                # Display predicted letter
                cv2.putText(frame, f'Predicted letter: {predicted_letter}', (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Calculate FPS
        end = time.time()
        fps = math.ceil(1 / (end - start))
        start = time.time()

        # Display FPS
        cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("capture image", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

        frames = list()

cv2.destroyAllWindows()
