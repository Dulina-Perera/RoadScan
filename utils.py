# utils.py

# %%
from numpy import ndarray
from typing import Dict, List, Tuple

# %%
def get_car(license_plate: List, vehicle_track_ids: ndarray) -> Tuple:
  """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
  """
  x1, y1, x2, y2, score, class_id = license_plate

  for i in range(len(vehicle_track_ids)):
    vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2, _ = vehicle_track_ids[i]

    if (x1 >= vehicle_x1 and y1 >= vehicle_y1 and x2 <= vehicle_x2 and y2 <= vehicle_y2):
      return vehicle_track_ids[i]

  return (-1, -1, -1, -1, -1)


def read_license_plate(license_plate_cropped: ndarray) -> Tuple:
  """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
  """

  return (0, 0)


def write_csv(results: Dict, output_path: str) -> None:
  """
		Write the results to a CSV file.

		Args:
				results (list): List of results to write to the CSV file.
				output_path (str): Path to the output CSV file.
	"""
  with open(output_path, 'w') as file:
    file.write(
      '{},{},{},{},{},{},{}\n'.format(
        'Frame_Number',
        'Vehicle_ID',
        'Vehicle_Bbox',
        'License_Plate_Bbox',
        'License_Plate_Bbox_Score',
        'License_Number',
        'License_Number_Score'
      )
    )

    for frame_no in results.keys():
      for vehicle_id in results[frame_no].keys():
        if ('vehicle' in results[frame_no][vehicle_id].keys() and
            'license_plate' in results[frame_no][vehicle_id].keys() and
            'text' in results[frame_no][vehicle_id]['license_plate'].keys()):
          file.write(
            '{},{},{},{},{},{},{}\n'.format(
							frame_no,
							vehicle_id,
							'[{} {} {} {}]'.format(
								results[frame_no][vehicle_id]['vehicle']['bbox'][0],
								results[frame_no][vehicle_id]['vehicle']['bbox'][1],
								results[frame_no][vehicle_id]['vehicle']['bbox'][2],
								results[frame_no][vehicle_id]['vehicle']['bbox'][3]
							),
							'[{} {} {} {}]'.format(
								results[frame_no][vehicle_id]['license_plate']['bbox'][0],
								results[frame_no][vehicle_id]['license_plate']['bbox'][1],
								results[frame_no][vehicle_id]['license_plate']['bbox'][2],
								results[frame_no][vehicle_id]['license_plate']['bbox'][3]
							),
							results[frame_no][vehicle_id]['license_plate']['bbox_score'],
							results[frame_no][vehicle_id]['license_plate']['text'],
							results[frame_no][vehicle_id]['license_plate']['text_score']
						)
					)

