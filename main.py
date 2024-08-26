# %%
import cv2
import numpy as np

from cv2 import VideoCapture
from lib.sort.sort import *
from numpy import ndarray
from typing import List
from ultralytics import YOLO

# %%
VEHICLES: List[int] = [2, 3, 5, 7]

tracker: Sort = Sort()

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
frame_no: int = -1
ret: bool = True
while ret:
  frame: ndarray
  ret, frame = cap.read()
  frame_no += 1

  if ret and frame_no < 10:
    # Detect vehicles.
    vehicles: ndarray = coco_model(frame)[0]
    vehicles_: List[List[float]] = []

    for detection in vehicles.boxes.data.tolist():
      x1: float; y1: float; x2: float; y2: float; score: float; class_id: float
      x1, y1, x2, y2, score, class_id = detection

      if int(class_id) in VEHICLES:
        vehicles_.append([x1, y1, x2, y2, score])

    # Track vehicles.
    track_ids: ndarray = tracker.update(np.asarray(vehicles_))

    # Detect license plates.
    license_plates: ndarray = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
      x1, y1, x2, y2, score, class_id = license_plate

      # Assign license plate to vehicle.

