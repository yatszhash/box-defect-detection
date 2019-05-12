import argparse

from scripts.resnet50_pretrained.holdout_train import main as resnet50_pretrained_train
from scripts.resnet50_pretrained_cosine_loss.holdout_train_dorpout_07_battch_64 import \
    main as resnet50_pretrained_cosine_train
from scripts.resnet50_pretrained_cosine_loss_pruned.holdout_train import main as resnet50_pretrained_cosine_pruned_train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    args = parser.parse_args()

    model_name = args["model"]
    if model_name == "resnet50_pretrained":
        resnet50_pretrained_train()
    elif model_name == "resnet50_pretrained_cosine":
        resnet50_pretrained_cosine_train()
    elif model_name == "resnet50_pretrained_cosine_pruned":
        resnet50_pretrained_cosine_pruned_train()
    else:
        raise ValueError(f"not supported model { model_name }")
