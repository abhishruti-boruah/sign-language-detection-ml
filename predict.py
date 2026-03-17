import cv2
import mediapipe as mp
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:

    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:

            data = []

            for lm in handLms.landmark:
                data.append(lm.x)
                data.append(lm.y)

            prediction = model.predict([data])

            cv2.putText(img, str(prediction[0]), (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Sign Detection", img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
