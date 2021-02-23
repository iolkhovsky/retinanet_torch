import torchvision
from bounding_box import bounding_box as bb


COLORS = ("navy", "blue", "aqua", "teal", "olive", "green", "lime", "yellow", "orange", "red", "maroon", "fuchsia",
          "purple", "black", "gray", "silver")
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


def visualize_object(in_img, input_bbox, label="", color="red"):
    img = in_img.copy()
    x, y, w, h = input_bbox
    x1, x2, y1, y2 = x, x + w - 1, y, y + h - 1
    bb.add(img, x1, y1, x2, y2, label, color)
    return img


def visualize_detection(img, class_id, class_label, conf, bbox):
    conf_str = "{:.2f}".format(conf)
    return visualize_object(img, bbox, label=f"{class_label} ({class_id}): {conf_str}")


def visualize_voc_annotation(img, voc_xml_annotation):
    out = img.copy()
    for obj in voc_xml_annotation["annotation"]["object"]:
        label = obj["name"]
        x = int(obj["bndbox"]["xmin"])
        y = int(obj["bndbox"]["ymin"])
        w = int(obj["bndbox"]["xmax"]) - x + 1
        h = int(obj["bndbox"]["ymax"]) - y + 1
        out = visualize_object(img, (x, y, w, h), label, color=VOC_CLASSES[label]["color"])
    return out


def build_voc2012(root="../data", subset="train"):
    return torchvision.datasets.VOCDetection(root=root, year="2012", image_set=subset, download=True)


