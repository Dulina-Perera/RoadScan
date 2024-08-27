from numpy import ndarray
from typing import List, Tuple


def read_license_plate(license_plate_cropped: ndarray) -> Tuple:
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    return (None, None)


def get_car(license_plate: List, vehicle_track_ids: ndarray) -> Tuple:
  """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
  """
  return (0, 0, 0, 0, 0)
