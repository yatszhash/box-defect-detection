import torchvision
from torch import nn


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
