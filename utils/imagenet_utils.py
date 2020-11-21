import json
import torch
from torchvision import transforms


def encode_id(idx, config_path):
    with open(config_path) as f:
        labels = json.load(f)
    return labels[idx]


def imagenet_classifier_inference(model, image, tensor_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                                  use_cuda=True):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(tensor_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    if torch.cuda.is_available() and use_cuda:
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    with torch.no_grad():
        output = model(input_batch)
    return torch.nn.functional.softmax(output[0], dim=0)


def imagenet_predict(model, image):
    out = imagenet_classifier_inference(model, image)
    idx = torch.argmax(out, axis=0)
    return idx, out[idx].item(), encode_id(idx, "/home/igor/github/my/retinanet_torch/configs/imagenet_classes.json")
