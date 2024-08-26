# %%
import cv2

from cv2 import VideoCapture
from numpy import ndarray
from typing import List
from ultralytics import YOLO

# %%
VEHICLES: List[int] = [2, 3, 5, 7]

# %%
# Load models.

# Model to detect vehicles.
coco_model: YOLO = YOLO(model='models/vehicle_detection/yolov8n.pt')

# Model to detect license plates.
license_plate_detector: YOLO = YOLO(model='./models/license_plate_detection/best_weights.pt')

# %%
# Load video.
cap: VideoCapture = VideoCapture('./static/videos/1.mp4')

# %%
# Read video frames.
frame_idx: int = -1
ret: bool = True
while ret:
  frame: ndarray
  ret, frame = cap.read()
  frame_idx += 1

  if ret and frame_idx < 10:
    # Detect vehicles.
    detections: ndarray = coco_model(frame)[0]
    detections_: List[List[float]] = []

    for detection in detections.boxes.data.tolist():
      x1: float; y1: float; x2: float; y2: float; score: float; class_id: float
      x1, y1, x2, y2, score, class_id = detection

      if int(class_id) in VEHICLES:
        detections_.append([x1, y1, x2, y2, score])
