# %%
import cv2
import numpy as np

from cv2 import VideoCapture
from lib.sort.sort import *
from numpy import ndarray
from typing import Dict, List
from ultralytics import YOLO
from utils import get_car, read_license_plate, write_csv

# %%
VEHICLES: List[int] = [2, 3, 5, 7]

tracker: Sort = Sort()

results: Dict = {}

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

  if ret:
    results[frame_no] = {}

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
      vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2, vehicle_id  = get_car(license_plate, track_ids)
      if vehicle_id != -1:
        # Crop license plate.
        license_plate_cropped: ndarray = frame[int(y1):int(y2), int(x1):int(x2)]

				# Process license plate.
        license_plate_cropped_gray: ndarray = cv2.cvtColor(license_plate_cropped, cv2.COLOR_BGR2GRAY)

        _: float; license_plate_cropped_thresh: ndarray
        _, license_plate_cropped_thresh = cv2.threshold(license_plate_cropped_gray, 64, 255, cv2.THRESH_BINARY_INV)

				# Read license plate.
        license_plate_text, license_plate_text_score = read_license_plate(license_plate_cropped_thresh)

        if license_plate_text is not None:
          results[frame_no][vehicle_id] = {
						'vehicle': {
							'bbox': [vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2]
						},
						'license_plate': {
							'bbox': [x1, y1, x2, y2],
							'bbox_score': score,
							'text': license_plate_text,
							'text_score': license_plate_text_score
						}
					}

# Write results.
write_csv(results, './results.csv')
