from models.resnet import resnet18
from models.mobilenetv2 import mobilenet_v2
from models.simple_cnn import SimpleCNN


def create_model(name, num_classes=10):
    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)

    elif name == "resnet18":
        return resnet18(num_classes=num_classes)

    elif name == "mobilenet_v2":
        return mobilenet_v2(num_classes=num_classes)

    else:
        raise ValueError(f"Unknown model: {name}")
