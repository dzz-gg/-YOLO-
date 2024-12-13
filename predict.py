import torch
import cv2
import sys

from ultralytics import YOLO


model = YOLO("../runs/detect/train/weights/best.pt")
results = model.predict(source=r'D:\Project\YOLOv8\dataset\valid\images\crazing_1.jpg')
annotated_frame = results[0].plot()
cv2.imshow("YOLOv8 Inference", annotated_frame)
cv2.waitKey()
cv2.destroyAllWindows()


