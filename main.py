import cv2 as cv
import time
import datetime

cap = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_fullbody.xml")

recording = True

while True:
    _, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = body_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) + len(bodies) > 0:
        recording = True

    # for (x,y,width,height) in faces:
    #     cv.rectangle(frame,(x,y),(x+width,y+height),(255,0,0),3)
    # for (x,y,width,height) in bodies:
    #     cv.rectangle(frame,(x,y),(x+width,y+height),(0,255,0),3)

    cv.imshow("Camera", frame)

    if cv.waitKey(1) == ord("q"):
        break


cap.release
cv.destroyAllWindows()    