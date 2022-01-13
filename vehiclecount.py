import cv2
import numpy as np

bike_cascade_path = r"two_wheeler.xml"
bike_cascade = cv2.CascadeClassifier()
if not bike_cascade.load(cv2.samples.findFile(bike_cascade_path)):
    print("bike cascade file not found")
    exit(0)

videopath = r"bikes.mp4"
cap = cv2.VideoCapture(videopath)
if not cap.isOpened:
    print("video capture failed")

count = 0
yval = 500

while True:
    ret, frame = cap.read()
    if frame is None:
        continue
    frame = np.array(frame)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    frame = cv2.line(frame, (0,yval), (len(frame[0]), yval), (0,0,255), 1)

    bikes = bike_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in bikes:
        frame = cv2.rectangle(frame, [x,y],[x+w,y+h], (0,0,255),4)
        if y+h/2 >= yval and y+h/2 <=yval+10:
            count+=1

    cv2.putText(frame, "Bike counter: {}".format(count), (0,len(frame)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)

    cv2.imshow('output',frame)
    if cv2.waitKey(1) & 0xFF == ord('q' or 'Q'):
        cv2.destroyAllWindows()
        break