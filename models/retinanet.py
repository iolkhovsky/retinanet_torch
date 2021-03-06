import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision

from dataset.utils import build_voc2012_for_ssd300, collate_voc2012
from models.mobilenet_v2 import SSDBackboneMobilenetv2
from models.ssd_predictor import RetinaNetPredictor
from models.bbox_codec import FasterRCNNBoxCoder
from models.anchor_generator import AnchorGenerator
from models.retinanet_loss import RetinaNetLoss
from utils.transforms import *
from utils.visualization import visuzalize_detection


class RetinanetMobilenet2(nn.Module):

    def __init__(self, anchors_cnt=6, classes_cnt=21):
        super(RetinanetMobilenet2, self).__init__()
        self.anchors, self.classes = anchors_cnt, classes_cnt
        self.backbone = SSDBackboneMobilenetv2(alpha=1., pretrained=True, requires_grad=True)
        self.predictor_heads = nn.ModuleList(
            [
                RetinaNetPredictor(in_channels=32, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt),
                RetinaNetPredictor(in_channels=96, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt),
                RetinaNetPredictor(in_channels=320, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt),
                RetinaNetPredictor(in_channels=480, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt),
                RetinaNetPredictor(in_channels=640, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt),
                RetinaNetPredictor(in_channels=640, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt, kernel=1, pad=0)
            ]
        )

    def forward(self, x):
        feature_maps = self.backbone(x)
        return [predictor(feature_map) for predictor, feature_map in zip(self.predictor_heads, feature_maps)]

    def __str__(self):
        feature_maps = len(self.predictor_heads)
        return f"Retinanet_Mobilenetv2_{feature_maps}fm_{self.classes}c_{self.anchors}a"


def visualize_prediction_target(inputs, targets, detections, dataformats='CHW', to_tensors=True, conf_thresh=5e-2):
    target_device = "cpu"

    target_imgs, predicted_imgs = [], []
    for img_idx, input_img in enumerate(inputs):
        if isinstance(input_img, torch.Tensor):
            if input_img.device != target_device:
                input_img = input_img.to(target_device)
            input_img = input_img.detach().numpy()

        input_img = ndarray_cyx2yxc(input_img)
        input_img = denormalize_image(input_img)
        input_img = (input_img * 255).astype(np.uint8)

        img = input_img.copy()
        for box, label in zip(targets[img_idx]["boxes"], targets[img_idx]["labels"]):
            if isinstance(box, torch.Tensor):
                if box.device != target_device:
                    box = box.to(target_device)
                box = box.detach().numpy()
            if isinstance(label, torch.Tensor):
                if label.device != target_device:
                    label = label.to(target_device)
                label = label.detach().numpy()
            img = visuzalize_detection(img, label=label, bbox=box, prob=1.0)
        target_imgs.append(img)

        img_logits, img_boxes = detections[img_idx]
        img_logits, img_boxes = torch.stack(img_logits), torch.stack(img_boxes)
        img_scores = F.softmax(img_logits, dim=1)
        max_scores, img_labels = torch.max(img_scores, dim=1)
        positive_detections_mask = torch.logical_and(max_scores >= conf_thresh, img_labels > 0)
        positive_cnt = torch.sum(positive_detections_mask.int())
        if positive_cnt == 0:
            img = input_img.copy()
            predicted_imgs.append(img)
            continue

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
        predicted_imgs.append(img)

    if dataformats == "CHW":
        target_imgs = [ndarray_yxc2cyx(x) for x in target_imgs]
        predicted_imgs = [ndarray_yxc2cyx(x) for x in predicted_imgs]

    if to_tensors:
        target_imgs = [torch.as_tensor(x) for x in target_imgs]
        predicted_imgs = [torch.as_tensor(x) for x in predicted_imgs]

    return target_imgs, predicted_imgs


class RetinanetLightning(pl.LightningModule):

    def __init__(self, classes_cnt=21, train_batch=2, val_batch=16, dataset_storage=None):
        super().__init__()
        aspect_ratios = [0.5, 1., 2.]
        scales = [2 ** x for x in [0, 1. / 3., 2. / 3.]]
        anchors_cnt = len(aspect_ratios) * len(scales)
        self.classes_cnt = classes_cnt
        self.model = RetinanetMobilenet2(anchors_cnt=anchors_cnt, classes_cnt=classes_cnt)
        self.box_coder = FasterRCNNBoxCoder()
        self.criterion = RetinaNetLoss(box_codec=self.box_coder, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt)
        self.anchor_gen = AnchorGenerator(aspect_ratios=aspect_ratios, scales=scales)
        self.anchors = []
        img_size = 300
        for map_size in [38, 19, 10, 5, 3, 1]:
            self.anchors.append(self.anchor_gen.generate(img_size, map_size))
        self.all_anchors = [torch.as_tensor(anchor) for map_anchors in self.anchors for anchor in map_anchors]
        self.all_anchors = torch.stack(self.all_anchors).to(self.device)
        self.max_predictions_per_map = 100
        self.iteration_idx = 0
        self.train_batch_size = train_batch
        self.val_batch_size = val_batch
        self.data_storage = dataset_storage
        self.train_dataset = None
        self.val_dataset = None
        self.input_shape = (300, 300)

    def prepare_data(self):
        self.train_dataset = build_voc2012_for_ssd300(root=self.data_storage, subset="train")
        self.val_dataset = build_voc2012_for_ssd300(root=self.data_storage, subset="val")
        """ TODO - Problem with logging graph with multiple outputs
        if self.logger.experiment:
            sample_input = torch.rand((1, 3, 300, 300))
            self.logger.experiment.add_graph(self.model, sample_input)"""

    def predict(self, x):
        assert len(x.size()) == 4
        predictions = self.model(x)
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

    def decode_prediction(self, x):
        batch_size = len(x)
        detections = [([], []) for _ in range(batch_size)]
        for img_idx, img_predictions in enumerate(x):
            for fmap_predictions, anchors in zip(img_predictions, self.anchors):
                clf_pred, rgr_pred = fmap_predictions
                assert len(clf_pred) == len(rgr_pred) == len(anchors)
                anchors = torch.as_tensor(anchors).to(self.device)

                boxes = self.box_coder.decode(rgr_pred, anchors)
                detections[img_idx][0].extend(clf_pred)
                detections[img_idx][1].extend(boxes)

        return detections

    def compute_loss(self, batch):
        inputs, targets = batch
        raw_prediction = self.predict(inputs)
        detections = self.decode_prediction(raw_prediction)

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
        batch_regr_loss /= batch_size
        return batch_loss, batch_clf_loss, batch_regr_loss, detections

    def forward(self, x):
        raw_prediction = self.predict(x)
        detections = self.decode_prediction(raw_prediction)
        return detections

    def postprocess(self, x, conf_threshold=0.1, nms=True):

        result = []
        for img_idx, (img_logits, img_boxes) in enumerate(x):

            img_dict = {"labels": [], "scores": [], "bboxes": []}
            img_logits, img_boxes = torch.stack(img_logits), torch.stack(img_boxes)
            img_scores = F.softmax(img_logits, dim=1)
            max_scores, img_labels = torch.max(img_scores, dim=1)
            positive_detections_mask = torch.logical_and(max_scores >= conf_threshold, img_labels > 0)
            positive_cnt = torch.sum(positive_detections_mask.int())
            if positive_cnt == 0:
                result.append(img_dict)
                continue

            max_scores = max_scores[positive_detections_mask]
            img_labels = img_labels[positive_detections_mask]
            img_boxes = img_boxes[positive_detections_mask]

            predicted_scores, predicted_labels, predicted_boxes = zip(*sorted(zip(max_scores, img_labels, img_boxes),
                                                                              reverse=True,
                                                                              key=lambda x: x[0]))
            predicted_scores = torch.stack(predicted_scores).detach().numpy()
            predicted_labels = torch.stack(predicted_labels).detach().numpy()
            predicted_boxes = torch.stack(predicted_boxes).detach().numpy()

            img_dict["labels"] = predicted_labels
            img_dict["scores"] = predicted_scores
            img_dict["bboxes"] = predicted_boxes

            result.append(img_dict)

        if nms:
            result = [self.non_max_suppression(img_dict) for img_dict in result]

        return result

    @staticmethod
    def non_max_suppression(detection_dict, iou_thresh=0.5):
        
        return detection_dict

    @staticmethod
    def preprocess_frame(cv_img):
        assert isinstance(cv_img, np.ndarray)
        assert len(cv_img.shape) == 3
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

    def visualize(self, cv_img, detections_dict, img_shape=None):
        img = cv_img.copy()
        detections_cnt = len(detections_dict["labels"])
        if img_shape is None:
            img_shape = self.input_shape
        ky, kx = img_shape[0] / self.input_shape[0], img_shape[1] / self.input_shape[1]
        src_img_boxes = []
        for x, y, w, h in detections_dict["bboxes"]:
            src_img_boxes.append((x * kx, y * ky, w * kx, h * ky))
        for detection_idx in range(detections_cnt):
            img = visuzalize_detection(img,
                                       label=detections_dict["labels"][detection_idx],
                                       bbox=src_img_boxes[detection_idx],
                                       prob=detections_dict["scores"][detection_idx],
                                       color=(0, 0, 255))
        return img

    def training_step(self, batch, batch_idx):
        batch = self.batch_to_device(batch, self.device)

        total, clf, regr, _ = self.compute_loss(batch)
        if self.logger.experiment:
            tboard = self.logger.experiment
            tboard.add_scalar("Loss/TrainTotal", total, global_step=self.iteration_idx)
            tboard.add_scalar("Loss/TrainClassification", clf, global_step=self.iteration_idx)
            tboard.add_scalar("Loss/TrainRegression", regr, global_step=self.iteration_idx)
        self.log("train_loss", total)
        self.iteration_idx += 1
        return total

    def validation_step(self, batch, batch_idx):
        batch = self.batch_to_device(batch, self.device)

        total, clf, regr, detections = self.compute_loss(batch)

        if self.logger.experiment:
            tboard = self.logger.experiment
            tboard.add_scalar("Loss/ValTotal", total, global_step=self.iteration_idx)
            tboard.add_scalar("Loss/ValClassification", clf, global_step=self.iteration_idx)
            tboard.add_scalar("Loss/ValRegression", regr, global_step=self.iteration_idx)
        self.log("val_loss", total)

        inputs, targets = batch
        target_imgs, pred_imgs = visualize_prediction_target(inputs, targets, detections, dataformats='CHW',
                                                             to_tensors=True, conf_thresh=0.1)
        img_grid_pred = torchvision.utils.make_grid(pred_imgs)
        img_grid_tgt = torchvision.utils.make_grid(target_imgs)
        if self.logger.experiment:
            tboard = self.logger.experiment
            tboard.add_image('Valid/Predicted', img_tensor=img_grid_pred, global_step=self.iteration_idx,
                             dataformats='CHW')
            tboard.add_image('Valid/Target', img_tensor=img_grid_tgt, global_step=self.iteration_idx,
                             dataformats='CHW')
        return total

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, collate_fn=collate_voc2012)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, collate_fn=collate_voc2012)

    def batch_to_device(self, batch, device):
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = [{"boxes": x["boxes"].to(device), "labels": x["labels"].to(device)} for x in targets]
        return inputs, targets

    def __str__(self):
        return f"Lightning_{self.model}"
