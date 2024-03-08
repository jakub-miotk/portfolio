import cv2
from itertools import compress


def get_non_overlapping_rects(rects1, rects2):
    """Get rectangles from rects1 which do not overlap with rectangles from rects2."""
    if len(rects1) < 1:
        return []
    if len(rects2) < 1:
        return rects1
    gr = cv2.groupRectangles(rects1 + rects2, 1, 0.5)[0]
    if len(gr) < 1:  # Condition is True if no rectangles are overlapping
        return rects1
    grouped_rects = list(zip(gr[:, 0], gr[:, 1], gr[:, 2], gr[:, 3]))
    # Get rects from rects1 that do not overlap grouped rects. This way less comparisons are required than
    # checking if a rect from rects1 overlaps any of rects from rects2
    is_not_overlapping = list(map(is_rect_not_overlapping_rects, rects1, [grouped_rects] * len(rects1)))
    return list(compress(rects1, is_not_overlapping))


def is_rect_not_overlapping_rects(rect, rects):
    for other_rect in rects:
        if rect_overlap(rect, other_rect):
            return False
    return True


def is_rect_within(outer_rect, inner_rect):
    xo, yo, wo, ho = outer_rect
    xi, yi, wi, hi = inner_rect
    return xo <= xi and yo <= yi and wo >= wi and ho >= hi


def merge_overlapping_rects(rects):
    if len(rects) == 0:
        return []
    rects.extend(rects.copy())
    gr = cv2.groupRectangles(rects, 1, 0.5)[0]
    return list(zip(gr[:, 0], gr[:, 1], gr[:, 2], gr[:, 3]))


def rect_overlap(rect1, rect2):
    x11, y11, w1, h1 = rect1
    x21, y21, w2, h2 = rect2
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2
    points_r1 = [(x11, y11), (x11, y12), (x12, y11), (x12, y12)]
    points_r2 = [(x21, y21), (x21, y22), (x22, y21), (x22, y22)]
    for p1 in points_r1:
        if (x21 <= p1[0] <= x22) and (y21 <= p1[1] <= y22):
            return True
    for p2 in points_r2:
        if (x11 <= p2[0] <= x12) and (y11 <= p2[1] <= y12):
            return True
    return False


def scale_rect(x, y, w, h, scale, max_w, max_h):
    """Rectangle is scaled proportionally to the scale value, but will not exceed set bounds."""
    w_s = int(w * scale)
    h_s = int(h * scale)
    x_s = x - int((w_s - w) / 2)
    y_s = y - int((h_s - h) / 2)
    if x_s < 0:
        x_s = 0
    if y_s < 0:
        y_s = 0
    if w_s >= max_w + x_s:
        w_s = max_w + x_s - 1
    if h_s >= max_h + y_s:
        h_s = max_h + y_s - 1
    return x_s, y_s, w_s, h_s
