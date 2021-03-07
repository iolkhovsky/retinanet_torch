import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision

from dataset.utils import build_voc2012_for_ssd300, collate_voc2012
from models.mobilenet_v2 import SSDBackboneMobilenetv2
from models.ssd_predictor import SSDPredictor
from models.bbox_codec import FasterRCNNBoxCoder
from models.anchor_generator import AnchorGenerator
from models.retinanet_loss import RetinaNetLoss
from utils.transforms import *
from utils.visualization import visuzalize_detection


class SSDMobilenet2(nn.Module):

    def __init__(self, anchors_cnt=6, classes_cnt=21):
        super(SSDMobilenet2, self).__init__()
        self.anchors, self.classes = anchors_cnt, classes_cnt
        self.backbone = SSDBackboneMobilenetv2(alpha=1., pretrained=True, requires_grad=True)
        self.predictor_heads = [
            SSDPredictor(in_channels=32, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt),
            SSDPredictor(in_channels=96, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt),
            SSDPredictor(in_channels=320, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt),
            SSDPredictor(in_channels=480, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt),
            SSDPredictor(in_channels=640, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt),
            SSDPredictor(in_channels=640, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt, kernel=1, pad=0)
        ]

    def forward(self, x):
        feature_maps = self.backbone(x)
        return [predictor(feature_map) for predictor, feature_map in zip(self.predictor_heads, feature_maps)]

    def __str__(self):
        feature_maps = len(self.classification_heads)
        return f"SSD_Mobilenetv2_{feature_maps}fm_{self.classes}c_{self.anchors}a"


def visualize_prediction_target(inputs, targets, detections, dataformats='CHW', to_tensors=True, conf_thresh=0.01):
    if type(detections) == torch.Tensor:
        detections = detections.detach().numpy()
    if type(targets) == torch.Tensor:
        targets = targets.detach().numpy()
    if type(inputs) == torch.Tensor:
        inputs = inputs.detach().numpy()

    target_imgs, predicted_imgs = [], []
    for img_idx, input_img in enumerate(inputs):
        input_img = ndarray_cyx2yxc(input_img)
        input_img = denormalize_image(input_img)
        input_img = (input_img * 255).astype(np.uint8)

        img = input_img.copy()
        for box, label in zip(targets[img_idx]["boxes"], targets[img_idx]["labels"]):
            label = label.item()
            bbox = box.numpy()
            img = visuzalize_detection(img, label=label, bbox=bbox, prob=1.0)
        target_imgs.append(img)

        img_logits, img_boxes = detections[img_idx]
        img_logits, img_boxes = torch.stack(img_logits), torch.stack(img_boxes)
        img_scores = F.softmax(img_logits, dim=1)
        max_scores, _ = torch.max(img_scores[:, 1:], dim=1)
        positive_mask = max_scores > conf_thresh
        positive_cnt = torch.sum(positive_mask.int())
        if positive_cnt == 0:
            img = input_img.copy()
            predicted_imgs.append(img)
            continue

        img_scores, img_boxes = img_scores[positive_mask], img_boxes[positive_mask]
        max_scores = max_scores[positive_mask]
        img_labels = torch.argmax(img_scores, dim=1)

        predicted_scores, predicted_labels, predicted_boxes = zip(*sorted(zip(max_scores, img_labels, img_boxes),
                                                                             reverse=True,
                                                                             key=lambda x: x[0]))

        img = input_img.copy()
        for score, label, bbox in zip(predicted_scores, predicted_labels, predicted_boxes):
            prob = score.item()
            label = label.item()
            bbox = bbox.numpy()
            img = visuzalize_detection(img, label=label, bbox=bbox, prob=prob, color=(0, 0, 255))
        predicted_imgs.append(img)

    if dataformats == "CHW":
        target_imgs = [ndarray_yxc2cyx(x) for x in target_imgs]
        predicted_imgs = [ndarray_yxc2cyx(x) for x in predicted_imgs]

    if to_tensors:
        target_imgs = [torch.as_tensor(x) for x in target_imgs]
        predicted_imgs = [torch.as_tensor(x) for x in predicted_imgs]

    return target_imgs, predicted_imgs


class SSDLightning(pl.LightningModule):

    def __init__(self, classes_cnt=21, tboard_writer=None):
        super().__init__()
        aspect_ratios = [0.5, 1., 2.]
        scales = [2 ** x for x in [0, 1. / 3., 2. / 3.]]
        anchors_cnt = len(aspect_ratios) * len(scales)
        self.classes_cnt = classes_cnt
        self.ssd = SSDMobilenet2(anchors_cnt=anchors_cnt, classes_cnt=classes_cnt)
        self.box_coder = FasterRCNNBoxCoder()
        self.criterion = RetinaNetLoss(box_codec=self.box_coder, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt)
        self.anchor_gen = AnchorGenerator(aspect_ratios=aspect_ratios, scales=scales)
        self.anchors = []
        img_size = 300
        for map_size in [38, 19, 10, 5, 3, 1]:
            self.anchors.append(self.anchor_gen.generate(img_size, map_size))
        self.all_anchors = [torch.as_tensor(anchor) for map_anchors in self.anchors for anchor in map_anchors]
        self.all_anchors = torch.stack(self.all_anchors)
        self.max_predictions_per_map = 100
        self.tboard = tboard_writer
        self.iteration_idx = 0

    def predict(self, x):
        assert len(x.size()) == 4
        predictions = self.ssd(x)
        assert predictions is not None
        batch_size = x.shape[0]
        inference_output = [[] for _ in range(batch_size)]

        for head_idx, head_prediction in enumerate(predictions):
            clf_pred, rgr_pred = head_prediction
            for imd_idx in range(batch_size):
                fmap_clf, fmap_rgr = clf_pred[imd_idx], rgr_pred[imd_idx]
                fmap_clf = fmap_clf.permute(1, 2, 0).reshape(-1, self.classes_cnt)
                fmap_rgr = fmap_rgr.permute(1, 2, 0).reshape(-1, 4)
                inference_output[imd_idx].append((fmap_clf, fmap_rgr))
        return inference_output

    def decode_output(self, x):
        assert isinstance(x, list)
        batch_size = len(x)
        detections = [([], []) for _ in range(batch_size)]
        for img_idx, img_predictions in enumerate(x):
            for fmap_predictions, anchors in zip(img_predictions, self.anchors):
                clf_pred, rgr_pred = fmap_predictions
                assert len(clf_pred) == len(rgr_pred) == len(anchors)
                max_confs, _ = torch.max(clf_pred[:, 1:], dim=1)
                anchors = torch.as_tensor(anchors)

                _, clf_pred, rgr_pred, anchors = zip(*sorted(zip(max_confs, clf_pred, rgr_pred, anchors),
                                                             reverse=True,
                                                             key=lambda x: x[0]))
                max_predictions = min(len(rgr_pred), self.max_predictions_per_map)
                clf_pred, rgr_pred, anchors = list(clf_pred), list(rgr_pred), list(anchors)
                clf_pred, rgr_pred, anchors = torch.stack(clf_pred), torch.stack(rgr_pred), torch.stack(anchors)

                boxes = self.box_coder.decode(rgr_pred[:max_predictions], anchors[:max_predictions])
                detections[img_idx][0].extend(clf_pred[:max_predictions])
                detections[img_idx][1].extend(boxes)

        return detections

    def feed_forward(self, x):
        raw_predictions = self.predict(x)
        batch_size = len(raw_predictions)
        detections = [([], []) for _ in range(batch_size)]
        for img_idx, img_predictions in enumerate(raw_predictions):
            for fmap_predictions, anchors in zip(img_predictions, self.anchors):
                clf_pred, rgr_pred = fmap_predictions
                assert len(clf_pred) == len(rgr_pred) == len(anchors)
                anchors = torch.as_tensor(anchors)

                boxes = self.box_coder.decode(rgr_pred, anchors)
                detections[img_idx][0].extend(clf_pred)
                detections[img_idx][1].extend(boxes)

        return detections

    def forward(self, x):
        with torch.no_grad():
            raw_predictions = self.predict(x)
            detections = self.decode_output(raw_predictions)
        return detections

    def compute_loss(self, batch):
        inputs, targets = batch
        detections = self.feed_forward(inputs)

        batch_size = max(len(batch), 1)
        batch_loss, batch_clf_loss, batch_regr_loss = 0., 0., 0.
        for img_detections, img_targets in zip(detections, targets):
            pred_confidences, pred_boxes = img_detections
            target_boxes = img_targets["boxes"]
            target_labels = img_targets["labels"]
            total, classification, regression = self.criterion(classification_preds=pred_confidences,
                                                               boxes_preds=pred_boxes,
                                                               anchors=self.all_anchors,
                                                               target_boxes=target_boxes,
                                                               target_labels=target_labels)
            batch_loss += total
            batch_clf_loss += classification
            batch_regr_loss += regression
        batch_loss /= batch_size
        batch_clf_loss /= batch_size
        batch_regr_loss /= batch_loss
        return batch_loss, batch_clf_loss, batch_regr_loss, detections

    def training_step(self, batch, batch_idx):
        self.ssd.train()
        total, clf, regr, _ = self.compute_loss(batch)
        if self.tboard:
            self.tboard.add_scalar("Loss/TrainTotal", total, global_step=self.iteration_idx)
            self.tboard.add_scalar("Loss/TrainClassification", clf, global_step=self.iteration_idx)
            self.tboard.add_scalar("Loss/TrainRegression", regr, global_step=self.iteration_idx)
        self.iteration_idx += 1
        return total

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            self.ssd.eval()
            total, clf, regr, detections = self.compute_loss(batch)

            if self.tboard:
                self.tboard.add_scalar("Loss/ValTotal", total, global_step=self.iteration_idx)
                self.tboard.add_scalar("Loss/ValClassification", clf, global_step=self.iteration_idx)
                self.tboard.add_scalar("Loss/ValRegression", regr, global_step=self.iteration_idx)

            inputs, targets = batch
            target_imgs, pred_imgs = visualize_prediction_target(inputs, targets, detections, dataformats='CHW',
                                                                 to_tensors=True, conf_thresh=0.1)
            img_grid_pred = torchvision.utils.make_grid(pred_imgs)
            img_grid_tgt = torchvision.utils.make_grid(target_imgs)
            if self.tboard:
                self.tboard.add_image('Valid/Predicted', img_tensor=img_grid_pred, global_step=self.iteration_idx,
                                      dataformats='CHW')
                self.tboard.add_image('Valid/Target', img_tensor=img_grid_tgt, global_step=self.iteration_idx,
                                      dataformats='CHW')
            return total

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        return DataLoader(build_voc2012_for_ssd300(subset="train"), batch_size=2, collate_fn=collate_voc2012)

    def val_dataloader(self):
        return DataLoader(build_voc2012_for_ssd300(subset="val"), batch_size=16, collate_fn=collate_voc2012)
