import torch
from torch.utils.data import DataLoader

from dataset.utils import build_voc2012_for_ssd300, collate_voc2012
from models.retinanet import RetinanetMobilenet2
from models.bbox_codec import FasterRCNNBoxCoder
from models.anchor_generator import AnchorGenerator
from models.retinanet_loss import RetinaNetLoss


train_dataloader = DataLoader(build_voc2012_for_ssd300(subset="train"), batch_size=2, collate_fn=collate_voc2012)

aspect_ratios = [0.5, 1., 2.]
scales = [2 ** x for x in [0, 1. / 3., 2. / 3.]]
anchors_cnt = len(aspect_ratios) * len(scales)
classes_cnt = 21
ssd = RetinanetMobilenet2(anchors_cnt=anchors_cnt, classes_cnt=classes_cnt)
optimizer = torch.optim.Adam(ssd.parameters(), lr=1e-3)
box_coder = FasterRCNNBoxCoder()
criterion = RetinaNetLoss(box_codec=box_coder, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt)
anchor_gen = AnchorGenerator(aspect_ratios=aspect_ratios, scales=scales)
anchors = []
img_size = 300
for map_size in [38, 19, 10, 5, 3, 1]:
    anchors.append(anchor_gen.generate(img_size, map_size))
all_anchors = [torch.as_tensor(anchor) for map_anchors in anchors for anchor in map_anchors]
all_anchors = torch.stack(all_anchors)
max_predictions_per_map = 100


def predict(x, model, classes_count=21):
    assert len(x.size()) == 4
    predictions = model(x)
    assert predictions is not None
    batch_size = x.shape[0]
    inference_output = [[] for _ in range(batch_size)]

    for head_idx, head_prediction in enumerate(predictions):
        clf_pred, rgr_pred = head_prediction
        for imd_idx in range(batch_size):
            fmap_clf, fmap_rgr = clf_pred[imd_idx], rgr_pred[imd_idx]
            fmap_clf = fmap_clf.permute(1, 2, 0).reshape(-1, classes_count)
            fmap_rgr = fmap_rgr.permute(1, 2, 0).reshape(-1, 4)
            inference_output[imd_idx].append((fmap_clf, fmap_rgr))
    return inference_output


def feed_forward(x, model, det_anchors):
    raw_predictions = predict(x, model)
    batch_size = len(raw_predictions)
    out_detections = [([], []) for _ in range(batch_size)]
    for img_idx, img_predictions in enumerate(raw_predictions):
        for fmap_predictions, fmap_anchors in zip(img_predictions, det_anchors):
            clf_pred, rgr_pred = fmap_predictions
            assert len(clf_pred) == len(rgr_pred) == len(fmap_anchors)
            fmap_anchors = torch.as_tensor(fmap_anchors)

            boxes = box_coder.decode(rgr_pred, fmap_anchors)
            out_detections[img_idx][0].extend(clf_pred)
            out_detections[img_idx][1].extend(boxes)

    return out_detections


for batch_idx, batch in enumerate(train_dataloader):
    inputs, targets = batch
    detections = feed_forward(inputs, ssd, anchors)
    optimizer.zero_grad()

    batch_loss = 0.
    for img_detections, img_targets in zip(detections, targets):
        pred_confidences, pred_boxes = img_detections
        target_boxes = img_targets["boxes"]
        target_labels = img_targets["labels"]
        total, classification, regression = criterion(classification_preds=pred_confidences,
                                                      boxes_preds=pred_boxes,
                                                      anchors=all_anchors,
                                                      target_boxes=target_boxes,
                                                      target_labels=target_labels)
        print(total.item(), classification.item(), regression.item())
        batch_loss += total
    batch_loss /= len(batch)
    print(f"Batch loss: {batch_loss.item()}")
    batch_loss.backward()
    optimizer.step()


