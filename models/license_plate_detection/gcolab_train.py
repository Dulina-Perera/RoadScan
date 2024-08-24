# %%
# %pip install ultralytics

# %%
import os

from google.colab import drive  # type: ignore
from typing import Any
from ultralytics.models.yolo.model import YOLO

# %%
drive.mount('/content/gdrive')
ROOT_DIR: str = '/content/gdrive/My Drive/license_plate_detection/resources'

# %%
model: YOLO = YOLO('yolov8n.yaml', verbose=True)

# %%
results: Any | None = model.train(
  data=os.path.join(ROOT_DIR, 'gdrive_data.yaml'),
  epochs=10,
  batch=64,
  imgsz=640,
  device='cuda',
  workers=8,
  optimizer='adam',
  lr0=0.01,
  patience=3,
)

# %%
# %scp --recursive /content/runs '/content/gdrive/My Drive/license_plate_detection'
