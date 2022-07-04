# !conda activate tf-gpu2.5
import torch
import cv2 
import numpy as np

print("Loading Dependencies...",'/n/n')

## Load Model
best_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

print("Loaded Models.",'/n')

## Make Detection
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    results = best_model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cap.destroyAllWindows()

print("Released")