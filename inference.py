import cv2
import numpy as np
from torch.nn import functional as F

from models.retinanet import RetinanetLightning
from utils.transforms import *
from utils.visualization import *

checkpoint = "/home/igor/models_checkpoints/retina.ckpt"
model = RetinanetLightning.load_from_checkpoint(checkpoint_path=checkpoint)
model.eval()

cap = cv2.VideoCapture(0)


def preprocess_frame(cv_img):
    height, width, _ = cv_img.shape
    cv_img = cv2.resize(cv_img, (300, 300))
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    rgb_img = rgb_img.astype(np.float32)
    rgb_img = rgb_img * 1. / 255.
    rgb_img = normalize_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    cyx_img = ndarray_yxc2cyx(rgb_img)
    cyx_batch = np.expand_dims(cyx_img, axis=0)
    input_tensor = torch.from_numpy(cyx_batch)
    return input_tensor


def visualize(input, detections, conf_thresh=1e-2):
    target_device = "cpu"

    if isinstance(input, torch.Tensor):
        if input.device != target_device:
            input_img = input.to(target_device)
        input_img = input_img.detach().numpy()

    input_img = input_img[0]

    input_img = ndarray_cyx2yxc(input_img)
    input_img = denormalize_image(input_img)
    input_img = (input_img * 255).astype(np.uint8)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    img_logits, img_boxes = detections[0]
    img_logits, img_boxes = torch.stack(img_logits), torch.stack(img_boxes)
    img_scores = F.softmax(img_logits, dim=1)
    max_scores, img_labels = torch.max(img_scores, dim=1)
    positive_detections_mask = torch.logical_and(max_scores >= conf_thresh, img_labels > 0)
    positive_cnt = torch.sum(positive_detections_mask.int())
    if positive_cnt == 0:
        img = input_img.copy()
        return img

    max_scores = max_scores[positive_detections_mask]
    img_labels = img_labels[positive_detections_mask]
    img_boxes = img_boxes[positive_detections_mask]

    predicted_scores, predicted_labels, predicted_boxes = zip(*sorted(zip(max_scores, img_labels, img_boxes),
                                                                      reverse=True,
                                                                      key=lambda x: x[0]))

    img = input_img.copy()
    for score, label, bbox in zip(predicted_scores, predicted_labels, predicted_boxes):
        if isinstance(score, torch.Tensor):
            if score.device != target_device:
                score = score.to(target_device)
            score = score.detach().numpy()
        if isinstance(label, torch.Tensor):
            if label.device != target_device:
                label = label.to(target_device)
            label = label.detach().numpy()
        if isinstance(bbox, torch.Tensor):
            if bbox.device != target_device:
                bbox = bbox.to(target_device)
            bbox = bbox.detach().numpy()
        img = visuzalize_detection(img, label=label, bbox=bbox, prob=score, color=(0, 0, 255))
    return img


while True:
    ret, frame = cap.read()

    cv2.imshow('frame', frame)

    input = preprocess_frame(frame)
    detections = model.feed_forward(input)
    vis = visualize(input, detections)
    cv2.imshow('vis', vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
