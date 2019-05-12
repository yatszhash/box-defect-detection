import PIL
import torch

from models import create_pretrained_resnet50, ImagenetTransformers


class Inferrer(object):
    CLASS_MAP = ["NG", "OK"]
    MODEL_FACTORIES = {
        "resnet50_pretrained": lambda: create_pretrained_resnet50(
            **{
                "dropout_rate": 0.7,
                "inner_units": 16,
                "n_class": 1
            }
        ),
        "resnet50_pretrained_cosine": lambda: create_pretrained_resnet50(
            **{
                "dropout_rate": 0.3,
                "inner_units": 16,
                "n_class": 2
            }
        ),
        "resnet50_pretrained_cosine_pruned": lambda: create_pretrained_resnet50(
            **{
                "dropout_rate": 0.3,
                "inner_units": 16,
                "n_class": 2
            }
        )
    }

    MODEL_STATE_PATHS = {
        "resnet50_pretrained": "output/resnet50_pretrained/holdout/model",
        "resnet50_pretrained_cosine": "output/resnet50_pretrained_cosine/holdout/model",
        "resnet50_pretrained_cosine_pruned:": "output/resnet50_pretrained_cosine_pruned/model"
    }

    def __init__(self, model_name="resnet50_pretrained"):
        self.model = self.MODEL_FACTORIES[model_name]()

        if torch.cuda.is_available():
            state_dict = torch.load(self.MODEL_STATE_PATHS[model_name], map_location="cuda")
        else:
            state_dict = torch.load(self.MODEL_STATE_PATHS[model_name], map_location="cpu")
        self.model.load_state_dict(state_dict)

    def __call__(self, image_path):
        x = PIL.Image.open(image_path)
        x = ImagenetTransformers()(x)
        self.model.eval()
        x = self.model(x.reshape([1] + list(x.shape)))
        if x.shape[1] == 1:
            x = torch.sigmoid(x)
        elif x.shape[1] == 2:
            x = torch.softmax(x, dim=1)[:, 1:]
        else:
            raise AssertionError()
        class_index = x.ge(0.5).cpu().detach().numpy()[0, 0]
        return self.CLASS_MAP[class_index]
