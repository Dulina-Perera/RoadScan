# %%
import os
import wandb

from dotenv import load_dotenv
from ultralytics.models.yolo.model import YOLO
from wandb.integration.ultralytics import add_wandb_callback

# %%
# Load environment variables from the .env file.
load_dotenv()

# Initialize Weights and Biases.
wandb.login(key=os.getenv('WANDB_API_KEY'), verify=True)

# %%
# Initialize a Weights and Biases run.
wandb.init(project='license_plate_detection', job_type='training')

# Load the model.
model: YOLO = YOLO('yolov10n.yaml', verbose=True)

# Add Weights and Biases callback for Ultralytics.
add_wandb_callback(model, enable_model_checkpointing=True)

# %%
# Train and finetune the model.
model.train(
  project='license_plate_detection',
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

# Validate the model.
model.val()

# %%
# Finalize the Weights and Biases run.
wandb.finish()
