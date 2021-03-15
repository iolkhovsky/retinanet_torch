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

while True:
    ret, frame = cap.read()

    cv2.imshow('frame', frame)

    input = model.preprocess_frame(frame)
    detections = model.feed_forward(input)
    vis = model.visualize(input, detections)
    cv2.imshow('vis', vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
