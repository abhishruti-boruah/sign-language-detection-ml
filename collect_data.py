import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

label = input("Enter sign label: ")

file = open("dataset/data.csv", "a", newline="")
writer = csv.writer(file)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:

            row = []

            for lm in handLms.landmark:
                row.append(lm.x)
                row.append(lm.y)

            row.append(label)
            writer.writerow(row)

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Collecting Data", img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
