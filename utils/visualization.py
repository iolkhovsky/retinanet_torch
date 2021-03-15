import cv2
import numpy as np

from utils.bounding_box import intersect_bboxes


DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX
DEFAULT_FONT_SCALE = 0.3


def visuzalize_detection(cv_image, label, bbox, prob=1.0, color=(0, 255, 0), thickness=None, auto_thickness=True,
                         base_thickness=8):
    assert isinstance(cv_image, np.ndarray)
    assert cv_image.shape[2] == 3
    img_height, img_width = cv_image.shape[:2]
    img = cv_image.copy()
    assert len(bbox) == 4
    label = str(label)
    x, y, w, h = bbox
    bbox = int(x), int(y), int(w), int(h)
    if thickness is None:
        thickness = max(int(prob * base_thickness), 1) if auto_thickness else 1
    else:
        assert isinstance(thickness, (int, float))
        thickness = max(int(thickness), 1)

    x, y, w, h = intersect_bboxes(bbox, (0, 0, img_width, img_height))

    img = cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
    org_x, org_y = x - 7, y - 7
    origin = min(max(org_x, 0), img_width), min(max(org_y, 0), img_height)
    description = f"{label}: {prob}"
    img = cv2.putText(img, description, origin, DEFAULT_FONT, DEFAULT_FONT_SCALE, (0, 0, 255), 1, cv2.LINE_AA)
    return img
