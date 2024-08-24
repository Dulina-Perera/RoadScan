# %%
from ultralytics import YOLO

# %%
# Load models.

# Model to detect vehicles.
coco_model: YOLO = YOLO(
  model='models/vehicle_detection/yolov8n.pt',
  task='train',
  verbose=True
)

# Model to detect license plates.
# license_plate_detector: YOLO = YOLO(
# 	model='yolov5s.pt',
# 	task='train',
# 	verbose=True
# )
