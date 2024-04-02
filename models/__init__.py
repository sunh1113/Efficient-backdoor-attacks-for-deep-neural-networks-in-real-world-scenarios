from models.vgg import vgg16
from models.resnet import resnet18
from models.mobilenetv2 import mobilenetv2


MODELS = {
    'v16': vgg16,
    'r18': resnet18,
    'mv2': mobilenetv2,
}

def build_model(model_name, num_classes):
    assert model_name in MODELS.keys()
    model = MODELS[model_name](num_classes)
    return model

