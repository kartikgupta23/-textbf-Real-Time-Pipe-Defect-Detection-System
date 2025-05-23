from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import numpy

cap = cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,720)

model = YOLO()
classNames=["Dents","Scratch","Holes","Cracks","No Detection"]

prev_frame_time=0
new_frame_time=0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img,sttream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x1 , y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))
            conf = math.ceil((box.conf[0]*100))/100
            cls  = int(box.cls[0])

            cvzone.putTextRect((img, f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)),scale=1))
