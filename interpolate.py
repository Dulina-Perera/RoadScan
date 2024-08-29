# interpolate.py

# %%
# Import the required libraries.
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from typing import Any, Dict, List

# %%
def parse_bbox(bbox: object) -> List[float]:
	"""Parse the bounding box string into a list of floats."""
	return list(map(float, bbox.strip('[]').split(' ')))


def interpolate_bbox(df: pd.DataFrame) -> pd.DataFrame:
	interpolated_rows: List[Dict[str, Any]] = []

  # Iterate through each unique vehicle.
	for vehicle_id in df['Vehicle_ID'].unique():
    # Filter the dataframe for the current vehicle.
		vehicle_df: pd.DataFrame = df[df['Vehicle_ID'] == vehicle_id].copy()

    # Identify missing frames.
		full_frames: np.ndarray[np.int64] = np.arange(vehicle_df['Frame_Number'].min(), vehicle_df['Frame_Number'].max() + 1)
		missing_frames: np.ndarray[np.int64] = np.setdiff1d(full_frames, vehicle_df['Frame_Number'].tolist())

		if len(missing_frames) == 0:
			interpolated_rows.extend(vehicle_df.to_dict('records'))
			continue

		# Prepare interpolation for bounding boxes.
		vehicle_bbox_data: np.ndarray = np.array(vehicle_df['Vehicle_Bbox'].tolist())
		license_plate_bbox_data: np.ndarray = np.array(vehicle_df['License_Plate_Bbox'].tolist())

		f_vehicle_bbox: interp1d = interp1d(vehicle_df['Frame_Number'], vehicle_bbox_data, kind='linear', axis=0)
		f_license_plate_bbox: interp1d = interp1d(vehicle_df['Frame_Number'], license_plate_bbox_data, kind='linear', axis=0)

		# Interpolate for missing frames.
		interpolated_vehicle_bboxes: Any = f_vehicle_bbox(missing_frames)
		interpolated_license_plate_bboxes: Any = f_license_plate_bbox(missing_frames)

		for (i, frame_number) in enumerate(missing_frames):
      # Create a new row for the interpolated bounding boxes.
			interpolated_rows.append({
				'Frame_Number': frame_number,
				'Vehicle_ID': vehicle_id,
				'Vehicle_Bbox': interpolated_vehicle_bboxes[i].tolist(),
				'License_Plate_Bbox': interpolated_license_plate_bboxes[i].tolist(),
				'License_Plate_Bbox_Score': 0.0,
    		'License_Number': None,
				'License_Number_Score': 0.0
			})

		interpolated_rows.extend(vehicle_df.to_dict('records'))

  # Return the full dataframe
	return pd.DataFrame(interpolated_rows)

# %%
df = pd.read_csv('results.csv')

df['Vehicle_Bbox'] = df['Vehicle_Bbox'].apply(parse_bbox)
df['License_Plate_Bbox'] = df['License_Plate_Bbox'].apply(parse_bbox)

# Interpolate the bounding boxes.
interpolated_df: pd.DataFrame = interpolate_bbox(df)
interpolated_df.sort_values(by=['Vehicle_ID', 'Frame_Number'], inplace=True)
interpolated_df.to_csv('interpolated_results.csv', index=False)
