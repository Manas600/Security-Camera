import cv2 as cv
import time
import datetime

cap = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_fullbody.xml")

detection = False
detection_start_time = None
SECONDS_TO_RECORD_AFTER_DETECTION = 10

frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv.VideoWriter_fourcc(*"mp4v")
out = None

while True:
    _, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = body_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) + len(bodies) > 0:
        if not detection:
            detection = True
            detection_start_time = time.time()
            currentTime = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv.VideoWriter(f"{currentTime}.mp4", fourcc, 30, frame_size)
            print("Started recording")
            
    else:
        if detection:
            if time.time() - detection_start_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                out.release()
                print("Stopped recording")
                print(f"Video saved as {currentTime}")

    if detection:
        out.write(frame)

    cv.imshow("Camera", frame)

    if cv.waitKey(1) == ord("q"):
        break

if detection:
    out.release()

cap.release()
cv.destroyAllWindows()
