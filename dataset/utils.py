import torchvision

from utils.transforms import *
from utils.visualization import *


COLORS = ((255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0),
          (255, 0, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),
          (0, 255, 0), (0, 0, 255), (0, 0, 255), (0, 0, 255),
          (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255))
DEFAULT_COLOR = COLORS[9]
VOC_CLASSES = \
    {
        "background":   {"id": 0, "color": COLORS[15]},
        "aeroplane":    {"id": 1, "color": COLORS[0]},
        "bicycle":      {"id": 2, "color": COLORS[1]},
        "bird":         {"id": 3, "color": COLORS[2]},
        "boat":         {"id": 4, "color": COLORS[3]},
        "bottle":       {"id": 5, "color": COLORS[4]},
        "bus":          {"id": 6, "color": COLORS[5]},
        "car":          {"id": 7, "color": COLORS[6]},
        "cat":          {"id": 8, "color": COLORS[7]},
        "chair":        {"id": 9, "color": COLORS[8]},
        "cow":          {"id": 10, "color": COLORS[9]},
        "diningtable":  {"id": 11, "color": COLORS[10]},
        "dog":          {"id": 12, "color": COLORS[11]},
        "horse":        {"id": 13, "color": COLORS[12]},
        "motorbike":    {"id": 14, "color": COLORS[13]},
        "person":       {"id": 15, "color": COLORS[14]},
        "pottedplant":  {"id": 16, "color": COLORS[15]},
        "sheep":        {"id": 17, "color": COLORS[0]},
        "sofa":         {"id": 18, "color": COLORS[1]},
        "train":        {"id": 19, "color": COLORS[2]},
        "tvmonitor":    {"id": 20, "color": COLORS[3]}
    }


def visualize_detection(img, class_id, class_label, conf, bbox):
    conf_str = "{:.2f}".format(conf)
    return


def visualize_voc_annotation(img, voc_xml_annotation):
    out = img.copy()
    for obj in voc_xml_annotation["annotation"]["object"]:
        label = obj["name"]
        x = int(obj["bndbox"]["xmin"])
        y = int(obj["bndbox"]["ymin"])
        w = int(obj["bndbox"]["xmax"]) - x + 1
        h = int(obj["bndbox"]["ymax"]) - y + 1
        conf_str = "{:.2f}".format(1.0)
        out = visuzalize_detection(img, label=label, bbox=(x, y, w, h), prob=conf_str, color=VOC_CLASSES[label]["color"])
    return out


def parse_annotation(voc_annotation_dict, scale_x=1., scale_y=1.):
    label = voc_annotation_dict["name"]
    x = float(voc_annotation_dict["bndbox"]["xmin"])
    y = float(voc_annotation_dict["bndbox"]["ymin"])
    w = float(voc_annotation_dict["bndbox"]["xmax"]) - x + 1
    h = float(voc_annotation_dict["bndbox"]["ymax"]) - y + 1
    return label, VOC_CLASSES[label]["id"], (int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y))


def transform_voc_item_ssd300(input, target):
    width, height = input.size
    scale_x = 300. / width
    scale_y = 300. / height
    cv_img = pil_to_cv_img(input)
    cv_img = cv2.resize(cv_img, (300, 300))
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    rgb_img = rgb_img.astype(np.float32)
    rgb_img = rgb_img * 1./255.
    rgb_img = normalize_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    cyx_img = ndarray_yxc2cyx(rgb_img)
    input_tensor = torch.from_numpy(cyx_img)
    target_objects = [parse_annotation(x, scale_x, scale_y) for x in target["annotation"]["object"]]
    target_boxes = torch.from_numpy(np.asarray([x[2] for x in target_objects]))
    target_labels = torch.from_numpy(np.asarray([x[1] for x in target_objects]))
    return input_tensor, {"boxes": target_boxes, "labels": target_labels}


def build_voc2012_for_ssd300(root=None, subset="train"):
    if root is None:
        root = "../data"
    return torchvision.datasets.VOCDetection(root=root,
                                             year="2012",
                                             image_set=subset,
                                             download=True,
                                             transforms=transform_voc_item_ssd300)


def collate_voc2012(data):
    batch_inputs, batch_targets = [], []
    for input, targets_dict in data:
        batch_inputs.append(input)
        batch_targets.append(targets_dict)
    return torch.stack(batch_inputs), batch_targets
