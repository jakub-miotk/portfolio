import numpy as np
import cv2
from Rect import is_rect_within
from LPImage import image_padding, invert_binary_image


def check_contour_angles(contour, max_par_angle, max_perp_angle):
    """Check if top and bottom edge are parallel or close to being parallel. Also check right edge's angle to the
    'ground'."""
    contour = reorder_contour_points(contour)
    # Check if shape has width
    if contour[0, 0] - contour[2, 0] == 0:
        return False
    if contour[1, 0] - contour[3, 0] == 0:
        return False
    # Check angle between top and bottom edge
    slope_top = np.abs((contour[0, 1] - contour[2, 1]) / (contour[0, 0] - contour[2, 0]))
    slope_bottom = np.abs((contour[1, 1] - contour[3, 1]) / (contour[1, 0] - contour[3, 0]))
    if max_par_angle < np.arctan(np.abs(slope_top - slope_bottom)) / (1 + (slope_top * slope_bottom)):
        return False
    # Check angle between right edge and 'ground'
    if contour[2, 0] == contour[3, 0]:
        return True
    slope_right = np.abs((contour[2, 1] - contour[3, 1]) / (contour[2, 0] - contour[3, 0]))
    if (np.pi / 2) - max_perp_angle > np.arctan(slope_right):
        return False
    return True


def check_contour_proportions(contour, proportions, proportions_sigma):
    """Check contours width to height proportions. Also check if bottom and top edge are of similar size."""
    contour = reorder_contour_points(contour)
    height = np.abs(contour[3, 1] - contour[2, 1])  # Bottom right y - top right y
    width = np.abs(contour[1, 0] - contour[3, 0])  # Bottom left x - bottom right x
    width_top = np.abs(contour[2, 0] - contour[0, 0])  # Top left x - top right x
    if height == 0 or width == 0:
        return False
    if height > width:
        return False
    if ~(0.8 <= width_top / width <= 1.2):  # One edge should not be 20% longer/shorter than the other
        return False
    cond = False
    for prop in proportions:  # True if close to any proportion in the list
        cond = cond or (prop - proportions_sigma < (width / height) < prop + proportions_sigma)
    return cond


def get_contours(binary_image):
    """Get 60 largest contours from a binary image."""
    edged_image = cv2.Canny(binary_image, 1, 30, apertureSize=7, L2gradient=True)
    contours, _ = cv2.findContours(edged_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=cv2.contourArea, reverse=True)[:60]


def get_lp_contour(lp_image, apply_filter=True):
    """Input image must be in grayscale."""
    if apply_filter:  # Sometimes input image might be already filtered
        lp_image = cv2.bilateralFilter(lp_image, 5, 255, 255)
    lp_image = image_padding(lp_image, 1, 255)
    _, th_img = cv2.threshold(lp_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th_img = invert_binary_image(th_img)
    contours, _ = cv2.findContours(th_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        return None
    # Get the largest contour in the center, that is not touching image bounds
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contour = contours[0]
    for c in contours:
        if is_contour_in_center(lp_image, c) and not is_contour_touching_bounds(c, 0, 0, lp_image.shape[1], lp_image.shape[0]):
            contour = c
            break
    # Very small contour is most likely not a lp contour
    min_contour_area = 0.01 * lp_image.shape[0] * lp_image.shape[1]
    if cv2.contourArea(contour) < min_contour_area:
        return None
    approx_c = cv2.approxPolyDP(contour, 0.004 * cv2.arcLength(contour, True), True)
    return approx_c


def is_contour_close_to_horizontal_axis(image, contour, max_dist):
    """Value of max_dist designates percentage of image's height as distance, that the center of a contour can be away
    from the central horizontal axis to be considered being close.
    With help of: https://pyimagesearch.com/2016/02/01/opencv-center-of-contour/"""
    if len(contour) < 2:
        return False
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return False
    cy = int(m["m01"] / m["m00"])
    axis_height = int(image.shape[0] / 2)
    return abs(axis_height - cy) <= max_dist * axis_height


def is_contour_in_center(image, contour):
    """Is the center of the image inside the contour."""
    outer_rect = cv2.boundingRect(contour)
    h, w = image.shape[:2]
    inner_rect = (int(w/2), int(h/2), 0, 0)
    return is_rect_within(outer_rect, inner_rect)


def is_contour_touching_bounds(contour, x, y, h, w):
    r_c = np.reshape(contour, (-1, 2))
    min_x, max_x, min_y, max_y = np.min(r_c[:, 0]), np.max(r_c[:, 0]), np.min(r_c[:, 1]), np.max(r_c[:, 1])
    # Is touching: top edge or bottom edge or left edge or right edge
    return min_x <= x or max_x >= x + h - 1 or min_y <= y or max_y >= y + w - 1


def preprocess_contour_shape(contour):
    if not isinstance(contour, np.ndarray):
        contour = np.array(contour)
    if len(contour.shape) > 2:
        contour = contour.reshape((4, 2))
    return contour


def reorder_contour_points(contour):
    """Put contour points in order: [[top-left], [bottom-left], [top-right], [bottom-right]]"""
    sort_idx = np.argsort(contour[:, 0])
    contour = contour[sort_idx, :]
    if contour[0, 1] > contour[1, 1]:
        buffer = contour[0, :]
        contour[0, :] = contour[1, :]
        contour[1, :] = buffer
    if contour[2, 1] > contour[3, 1]:
        buffer = contour[2, :]
        contour[2, :] = contour[3, :]
        contour[3, :] = buffer
    return contour
