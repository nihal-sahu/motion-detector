import cv2, time, pandas
from datetime import datetime

first_frame = None
status_list = [None, None]
times = []
df = pandas.DataFrame(columns = ["Start", "End"])

video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    check, frame = video.read()
    status = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue
    
    faces = face_cascade.detectMultiScale(frame, 
    scaleFactor = 1.1,
    minNeighbors = 5)

    for x, y, w, h in faces:
        frame = cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0))
        status = 1
    
    status_list.append(status)
    
    if status_list[-1] == 1 and status_list[-2] == 0: 
        times.append(datetime.now())

    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        if status == 1: 
            times.append(datetime.now())
        break
    
print(status_list)
print(times)

for i in range(0, len(times), 2):
    df = df.append({"Start": times[i], "End": times[i + 1]}, ignore_index = True)

df.to_csv("Times.csv")

video.release()
cv2.destroyAllWindows()