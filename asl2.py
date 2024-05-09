import cv2
import mediapipe as mp
import math
import time
import tensorflow as tf
import os
import numpy as np
import predict

# Provide the path to the directory where the model is saved
model_path = os.curdir + "/asl_model.keras"

# Load the saved model
loaded_model = tf.keras.models.load_model(model_path)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

mp_drawig = mp.solutions.drawing_utils

mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

start = time.time()  # Set start

end = 1

predicted_letter = None

while True:

    success, frame = cap.read()
    if success:
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hand.process(RGB_frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                p_this = list()
                for id, lm in enumerate(hand_landmarks.landmark):
                    p_this.append(float(lm.x))
                    p_this.append(float(lm.y))
                    p_this.append(float(lm.z))

                print(p_this)

                p_this = np.array(p_this).reshape(1, 63)

                #print(p_this)
                #exit()

                predicted_letter = predict._predict(p_this, loaded_model)

                #print(index)
                # Draw landmark
                mp_drawig.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Calculate FPS
        end = time.time()
        fps = math.ceil(1 / (end - start))
        start = time.time()



        # Display FPS
        cv2.putText(frame, f'FPS: {fps}; Predicted letter: {predicted_letter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("capture image", frame)
        if cv2.waitKey(1) == ord('q'):
            break

        predicted_letter = None

cv2.destroyAllWindows()


