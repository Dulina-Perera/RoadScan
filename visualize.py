# visualize.py

# %%
import ast
import cv2
import numpy as np
import pandas as pd

from cv2 import VideoCapture, VideoWriter
from numpy import ndarray
from pandas import DataFrame
from typing import Dict, List

# %%
def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
  x1, y1 = top_left
  x2, y2 = bottom_right

  cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
  cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

  cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
  cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

  cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
  cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

  cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
  cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

  return img

# %%
if __name__ == '__main__':
  video_path: str = './static/videos/1.mp4'
  results: DataFrame = pd.read_csv('./interpolated_results.csv')

  cap: VideoCapture = VideoCapture(video_path)

  fourcc: int = cv2.VideoWriter_fourcc(*'mp4v')
  fps: float = cap.get(cv2.CAP_PROP_FPS)
  width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  out: VideoWriter = cv2.VideoWriter('./static/videos/1_out.mp4', fourcc, fps, (width, height))

	# Extract license number and the corresponding crop with the highest score.
  license_plates: Dict = {}
  for vehicle_id in results['Vehicle_ID'].unique():
    # Get the highest score and the corresponding license number.
    max_score: float = results[results['Vehicle_ID'] == vehicle_id]['License_Number_Score'].max()
    license_plates[vehicle_id] = {
			'license_plate_crop': None,
			'license_number': results[(results['Vehicle_ID'] == vehicle_id) & (results['License_Number_Score'] == max_score)]['License_Number'].iloc[0]
		}

		# Get the frame number and read the frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['Vehicle_ID'] == vehicle_id) & (results['License_Number_Score'] == max_score)]['Frame_Number'].iloc[0])
    ret, frame = cap.read()

    bbox = results[(results['Vehicle_ID'] == vehicle_id) & (results['License_Number_Score'] == max_score)]['License_Plate_Bbox']
    x1, y1, x2, y2 = ast.literal_eval(bbox.iloc[0])

    license_plate_crop: ndarray = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_plate_crop: ndarray = cv2.resize(license_plate_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

    license_plates[vehicle_id]['license_plate_crop'] = license_plate_crop

  frame_no: int = -1
  cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
  ret: bool = True
  while ret:
    frame: ndarray
    ret, frame = cap.read()
    frame_no += 1

    if ret:
      df_temp: DataFrame = results[results['Frame_Number'] == frame_no]
      for row_idx in range(len(df_temp)):
        vehicle_bbox: List[float] = df_temp.iloc[row_idx]['Vehicle_Bbox']
        license_plate_bbox: List[float] = df_temp.iloc[row_idx]['License_Plate_Bbox']

        # Draw vehicle.
        vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2 = ast.literal_eval(vehicle_bbox)
        draw_border(frame, (int(vehicle_x1), int(vehicle_y1)), (int(vehicle_x2), int(vehicle_y2)), (0, 255, 0), 12)

        # Draw license plate.
        x1, y1, x2, y2 = ast.literal_eval(license_plate_bbox)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

        # Display cropped license plate and license number.
        license_plate_crop: ndarray = license_plates[df_temp.iloc[row_idx]['Vehicle_ID']]['license_plate_crop']
        H, W, _ = license_plate_crop.shape

        try:
          frame[int(vehicle_y1) - H - 100:int(vehicle_y1) - 100, int((vehicle_x2 + vehicle_x1 - W) / 2):int((vehicle_x2 + vehicle_x1 + W) / 2), :] = license_plate_crop
          frame[int(vehicle_y1) - H - 400:int(vehicle_y1) - H - 100, int((vehicle_x2 + vehicle_x1 - W) / 2):int((vehicle_x2 + vehicle_x1 + W) / 2), :] = (255, 255, 255)

          (text_width, text_height), _ = cv2.getTextSize(license_plates[df_temp.iloc[row_idx]['Vehicle_ID']]['license_number'], cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17)
          cv2.putText(
          	frame,
          	license_plates[df_temp.iloc[row_idx]['Vehicle_ID']]['license_number'],
          	(
            	int((vehicle_x2 + vehicle_x1 - text_width) / 2), int(vehicle_y1 - H - 250 + (text_height / 2)),
  						cv2.FONT_HERSHEY_SIMPLEX,
           		4.3,
            	(0, 0, 0),
            	17
          	)
        	)
        except:
          pass

      out.write(frame)
      frame = cv2.resize(frame, (1280, 720))

out.release()
cap.release()
