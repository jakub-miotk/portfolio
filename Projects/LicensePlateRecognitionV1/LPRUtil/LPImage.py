import numpy as np
import cv2
from Contour import get_lp_contour, is_contour_close_to_horizontal_axis


def binarize(image):
    """Blur grayscale image with bilateral filter and apply a threshold."""
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.bilateralFilter(grayscale_image, 2, 20, 1)
    _, binary_image1 = cv2.threshold(grayscale_image, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, binary_image2 = cv2.threshold(grayscale_image, 70, 255, cv2.THRESH_BINARY)
    binary_image = binary_image1 + binary_image2
    return binary_image


def get_lp_skew_angle(lp_image, apply_filter=True):
    """Input image must be in grayscale."""
    if apply_filter:  # Sometimes input image might be already filtered
        lp_image = cv2.bilateralFilter(lp_image, 5, 255, 255)
    approx_c = get_lp_contour(lp_image, apply_filter=False)
    if approx_c is None:
        return 0
    if len(approx_c) < 1:
        return 0
    approx_c = cv2.boxPoints(cv2.minAreaRect(approx_c))
    approx_c = approx_c.reshape((-1, 1, 2)).astype(dtype=int)
    vx, vy, x, y = cv2.fitLine(approx_c, cv2.DIST_L2, 0, 0.01, 0.01)
    angle = np.arctan(vy / vx)[0]
    # Extreme angle suggests that something went wrong during contour extraction and in that case
    # no change should be done
    if np.abs(np.rad2deg(angle)) > 30:
        angle = 0
    return angle


def image_padding(image, px, bg_colour=255):
    """Add background padding to binary image. px can be a tuple (height, width) or a single value.
    Values can be integers or floats. Float values cause the background size to be percentage
    of the size of original image. Integer values cause height and width to be increased by 2*px each."""
    h, w = image.shape
    if isinstance(px, tuple):
        px_h, px_w = px
    else:
        px_h = px_w = px
    if isinstance(px_h, float):
        px_h = int(h * px_h / 2)
    if isinstance(px_w, float):
        px_w = int(w * px_w / 2)
    background = np.ones((h + (px_h * 2), w + (px_w * 2)), dtype=np.uint8) * bg_colour
    background[px_h:px_h + h, px_w:px_w + w] = image
    return background


def invert_binary_image(image):
    return cv2.bitwise_not(image)


def is_image_empty(image):
    if len(np.unique(image)) < 2:
        return False
    return True


def level_lp(lp_image, apply_filter=True):
    """Rotate license plate image to make it parallel to the ground."""
    angle = get_lp_skew_angle(lp_image, apply_filter=apply_filter)
    return rotate_img(lp_image, angle)


def prepare_lp_for_ocr(lp_image, ih, mh, mw, btb, blr):
    """This function extracts a license plate from an image and returns processed version ready for ocr.
    Parameter's full names: image height, margin width, margin height, border top/bottom, border left/right."""
    if len(lp_image.shape) > 2:
        lp_image = cv2.cvtColor(lp_image, cv2.COLOR_BGR2GRAY)
    lp_image = rescale_lp_by_height(lp_image, target_h=300)
    lp_image = cv2.bilateralFilter(lp_image, 5, 255, 255)
    lp_image = level_lp(lp_image, apply_filter=False)
    lp_image = threshold_lp(lp_image)
    lp_image = remove_non_lp_elements(lp_image)
    if lp_image is None:
        return None
    lp_image = remove_remaining_noise(lp_image)
    # Sometimes poor quality of a lp image can lead to entire image be treated as noise
    if lp_image.shape[0] == 0 or lp_image.shape[1] == 0:
        return None
    lp_image = rescale_lp_by_height(lp_image, ih)
    lp_image[lp_image < 128] = 0  # Removes non-binary colours added as a result of interpolation
    lp_image = image_padding(lp_image, (mw, mh), 255)  # Add white margin
    lp_image = image_padding(lp_image, (btb, blr), 0)  # Add black border
    return lp_image


def remove_background(image):
    """Find outermost rows and columns that have only background color values and crop them out."""
    if len(np.unique(image)) < 2:  #
        return image
    # Find indexes of first/last rows/columns with values equal 0 (other values being background)
    l_col, r_col = np.argwhere(np.min(image, axis=0) == 0)[[0, -1]].flatten()
    t_row, b_row = np.argwhere(np.min(image, axis=1) == 0)[[0, -1]].flatten()
    return image[t_row:b_row, l_col:r_col]


def remove_non_lp_elements(lp_image):
    """Remove all elements outside of license plate's white background."""
    lp_image = invert_binary_image(lp_image)
    lp_contour = get_lp_contour(lp_image, apply_filter=False)
    if lp_contour is None:  # Sometimes lp's contour cannot be properly separated
        return None
    lp_contour = cv2.boxPoints(cv2.minAreaRect(lp_contour))
    lp_contour = lp_contour.reshape((-1, 1, 2)).astype(dtype=int)
    mask = np.zeros_like(lp_image)
    mask = cv2.drawContours(mask, [lp_contour], -1, (255, 255, 255), cv2.FILLED)
    # Reduce mask's size to get rid of most of the noise present on the edges of a license plate.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (2, 2))
    mask = cv2.erode(mask, kernel, iterations=10)
    lp_image[mask == 0] = 0
    return invert_binary_image(lp_image)


def remove_remaining_noise(lp_image):
    """Remove contours far away from the central horizontal line and other very small ones."""
    # Removal of contours far away from the center
    contours, _ = cv2.findContours(lp_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_dist = 0.35
    contours = [contour for contour in contours if is_contour_close_to_horizontal_axis(lp_image, contour, max_dist)]
    mask = np.zeros_like(lp_image)
    mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), cv2.FILLED)
    lp_image[mask == 255] = 255
    # Removal of very small contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lp_image = cv2.dilate(lp_image, kernel, iterations=4)
    lp_image = cv2.erode(lp_image, kernel, iterations=4)
    return remove_background(lp_image)


def rescale_lp_by_height(lp_image, target_h=300):
    scale_factor = target_h / lp_image.shape[0]
    h, w = lp_image.shape[:2]
    h = int(h * scale_factor)
    w = int(w * scale_factor)
    return cv2.resize(lp_image, (w, h), interpolation=cv2.INTER_LINEAR)


def rotate_img(img, angle):
    """Input image must be in grayscale."""
    h, w = img.shape[:2]
    hm, wm = int(h * 1.5), int(2 * 1.5)  # height margin, width margin
    hm05, wm05 = int(hm * 0.5), int(wm * 0.5)
    img_m = np.ones((h + hm, w + wm), dtype=np.uint8) * 255  # Image with margin
    img_m[hm05:h + hm05, wm05:w + wm05] = img
    old_h, old_w = h, w
    h += hm
    w += wm
    rm2d = cv2.getRotationMatrix2D((w / 2, h / 2), np.rad2deg(angle), 1)
    img_m = cv2.warpAffine(img_m, rm2d, (w, h), borderMode=cv2.BORDER_REPLICATE)
    img_m = img_m[hm05:old_h + hm05, wm05:old_w + wm05]  # Remove margin
    ang_len = int(np.abs(np.tan(angle) * old_w * 0.45))
    if ang_len >= old_h * 0.5:  # Do not crop the rotated image if it would cause a loss of height greater than 50%
        ang_len = 0
    return img_m[ang_len:old_h - ang_len, :]


def threshold_lp(lp_image):
    """Binarize license plate image."""
    h, w = lp_image.shape
    per = 0.95  # Percentage of image to be ignored when determining threshold value
    hi = int(h * per / 2)
    wi = int(w * per / 2)
    # Only central fragment of an image is used to determine threshold, otherwise unnecessary background can have
    # an impact on the threshold value
    cutout_for_thresholding = lp_image[hi:h - hi, wi:w - wi]
    threshold, _ = cv2.threshold(cutout_for_thresholding, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, lp_image = cv2.threshold(lp_image, threshold, 255, cv2.THRESH_BINARY)
    return lp_image
