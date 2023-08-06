import cv2 as cv
import time
import datetime

cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()

    cv.imshow("Camera", frame)

    if cv.waitKey(1) == ord("q"):
        break


cap.release
cv.destroyAllWindows()    