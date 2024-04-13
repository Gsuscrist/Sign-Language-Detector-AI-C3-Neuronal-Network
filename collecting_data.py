# THIS FILE TAKE THE DATA TO TRAIN THE NN

import os
import cv2
import time

DATA_DIR = './data/images'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

classes = 26
dataset_size = 10

cap = cv2.VideoCapture(0)
for i in range(classes):
    letter = chr(ord('a') + i)
    if not os.path.exists(os.path.join(DATA_DIR, letter)):
        os.makedirs(os.path.join(DATA_DIR, letter))

    message = 'Collecting data for class {}'.format(letter)
    print(message)

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, message+' Ready? Press "Q" !', (25, 25), cv2.FONT_ITALIC, 0.8, (0, 0, 0), 1,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        cv2.waitKey(50)
        timestamp = int(time.time())
        cv2.imwrite(os.path.join(DATA_DIR, letter, '{}-{}_{}.jpg'.format(letter,counter, timestamp)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
