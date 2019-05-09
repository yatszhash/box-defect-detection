import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, RandomResizedCrop, RandomHorizontalFlip


def create_pretrained_resnet50(dropout_rate=0.4, inner_units=32, n_class=1):
    resnet = torchvision.models.resnet50(pretrained=True)
    for param in resnet.parameters():
        param.required_grad = False
    out_channel = resnet.fc.out_features
    return nn.Sequential(
        resnet,
        nn.Dropout(dropout_rate),
        nn.Linear(out_channel, inner_units),
        nn.Dropout(dropout_rate),
        nn.Linear(inner_units, n_class)
    )


class PretrainedResnet50WithClassEmbedding(nn.Module):

    def __init__(self, dropout_rate=0.4, inner_units=32, class_embedding=2, n_class=1):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.required_grad = False
        out_channel = resnet.fc.out_features
        self.resnet_classifier = nn.Sequential(
            resnet,
            nn.Dropout(dropout_rate),
            nn.Linear(out_channel, inner_units),
            nn.Dropout(dropout_rate),
            nn.Linear(inner_units, class_embedding),
        )
        self.last_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(class_embedding, n_class),
        )

    def forward(self, X):
        embedded_class = self.resnet_classifier(X)
        output = self.last_layer(embedded_class)

        return embedded_class, output


class ImagenetTransformers:
    SIZE = 224

    def __init__(self):
        transform_list = [
            # transforms.ToPILImage(),
            Resize(self.SIZE),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        self.transforms = Compose(transform_list)

    def __call__(self, *args, **kwargs):
        return self.transforms(*args, **kwargs)


class ImagenetAugmentTransformers:
    SIZE = 224

    def __init__(self):
        transform_list = [
            RandomResizedCrop(size=self.SIZE, scale=(0.85, 1.0), ratio=(1, 1)),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        self.transforms = Compose(transform_list)

    def __call__(self, *args, **kwargs):
        return self.transforms(*args, **kwargs)


class CosineLoss(nn.Module):

    def forward(self, inputs: torch.Tensor, target: torch.Tensor):
        if len(target.shape) == 1:
            target = F.one_hot(target.long()).float()
        elif len(target.shape) == 2 and target.shape[1] == 1:
            target = F.one_hot(target.long().reshape(-1)).float()
        normalized_input = inputs / inputs.norm(dim=1, keepdim=True)
        dot_product = (normalized_input * target).sum(dim=1)
        return (1 - dot_product).mean()


class CosineCrossEntropyLoss(nn.Module):

    def __init__(self, lambda_):
        super().__init__()
        self.cosine_loss = CosineLoss()
        self.lambda_ = lambda_

    def forward(self, class_embedded: torch.Tensor, inputs: torch.Tensor, target: torch.Tensor):
        target = target.long().reshape(-1)
        cosine_loss_value = self.cosine_loss(class_embedded, target)
        entropy_value = F.cross_entropy(inputs, target) if inputs.shape[1] > 1 else F.binary_cross_entropy_with_logits(
            inputs, target)
        return cosine_loss_value + self.lambda_ * entropy_value
