# utils.py

# %%
import string

from easyocr import Reader
from numpy import ndarray
from typing import Any, Dict, List, Tuple

# %%
reader: Reader = Reader(['en'])

# Mappings for character conversion.
CHAR_TO_INT: Dict[str, str] = {
	'O': '0',
	'I': '1',
	'J': '3',
	'A': '4',
	'S': '5',
	'G': '6'
}

INT_TO_CHAR: Dict[str, str] = {
	'0': 'O',
	'1': 'I',
	'3': 'J',
	'4': 'A',
	'5': 'S',
	'6': 'G'
}

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


def license_complies_with_format(text: str) -> bool:
  """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
  """
  if len(text) != 7:
    return False

  if (
		(text[0] in string.ascii_uppercase or text[0] in INT_TO_CHAR.keys()) and
		(text[1] in string.ascii_uppercase or text[1] in INT_TO_CHAR.keys()) and
		(text[2] in [str(i) for i in range(10)] or text[2] in CHAR_TO_INT.keys()) and
		(text[3] in [str(i) for i in range(10)] or text[3] in CHAR_TO_INT.keys()) and
		(text[4] in string.ascii_uppercase or text[4] in INT_TO_CHAR.keys()) and
		(text[5] in string.ascii_uppercase or text[5] in INT_TO_CHAR.keys()) and
		(text[6] in string.ascii_uppercase or text[6] in INT_TO_CHAR.keys())
	):
    return True
  else:
    return False


def format_license(text: str) -> str:
  """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
  """
  text_: str = ''
  mappings: Dict[int, Dict[str, str]] = {
		0: INT_TO_CHAR,
		1: INT_TO_CHAR,
		2: CHAR_TO_INT,
		3: CHAR_TO_INT,
		4: INT_TO_CHAR,
		5: INT_TO_CHAR,
		6: INT_TO_CHAR
	}

  for i in [0, 1, 2, 3, 4, 5, 6]:
    if text[i] in mappings[i].keys():
      text_ += mappings[i][text[i]]
    else:
      text_ += text[i]

  return text_


def read_license_plate(license_plate_cropped: ndarray) -> Tuple:
  """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
  """
  detections: Any = reader.readtext(license_plate_cropped)
  for detection in detections:
    _, text, score = detection

    text = text.upper().replace(' ', '')

    if license_complies_with_format(text):
      return (format_license(text), score)

  return (None, None)



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

