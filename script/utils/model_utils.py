import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, resnet18, resnet50, densenet169


def classifier_model(name='resnet50', pretrained=True,  num_classes=4, **kwargs):
    if name == 'densenet169':
        model = build_densenet169(pretrained=pretrained, num_classes=num_classes)
    elif name == 'resnet18':
        model = build_resnet18(pretrained=pretrained, num_classes=num_classes)
    elif name == 'resnet50':
        model = build_resnet50(pretrained=pretrained, num_classes=num_classes)
    elif name == 'vgg16':
        model = build_vgg16(pretrained=pretrained, num_classes=num_classes)
    return model

def build_densenet169(pretrained=True, num_classes=4):
    model = densenet169(pretrained=pretrained)
    in_features = model.classifier.in_features 
    model.classifier = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=num_classes),
    )
    return model

def build_resnet18(pretrained=True, num_classes=4):
    model = resnet18(pretrained=pretrained)
    in_features = model.fc.in_features 
    model.fc = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=num_classes),
    )
    return model

def build_resnet50(pretrained=True, num_classes=4):
    model = resnet50(pretrained=pretrained)
    in_features = model.fc.in_features 
    model.fc = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=num_classes),
    )
    return model

def build_vgg16(pretrained=True, num_classes=4):
    model = vgg16(pretrained=pretrained)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=num_classes),
    )
    return model

if __name__ == '__main__':
    import torchinfo
    net = build_densenet169()
    print(torchinfo.summary(net, input_size=(4,3,64,64),col_names=["output_size", "num_params"],))