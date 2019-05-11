from pathlib import Path

from scripts.base_evaluate import run

if __name__ == '__main__':
    model_dir = Path(__file__).parent.parent.parent.joinpath("output/resnet50_pretrained_fixed/holdout")
    params = {
        "dropout_rate": 0.7,
        "inner_units": 16,
        "n_class": 1
    }

    run(model_dir, params)
