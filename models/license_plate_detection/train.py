# %%
from typing import Any
from ultralytics.models.yolo.model import YOLO

# %%
# Load the model.
model: YOLO = YOLO('yolov8n.yaml', verbose=True)

# %%
results: Any | None = model.train(
  data='./resources/data.yaml',
  epochs=10,
  batch=64,
  imgsz=640,
  device='cuda',
  workers=8,
  optimizer='adam',
  lr0=0.01,
  patience=3,
)
