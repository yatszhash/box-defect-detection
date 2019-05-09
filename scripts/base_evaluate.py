from pathlib import Path

from box_dataset import BoxDataFolder
from model_wrapper import ModelEvaluation
from models import ImagenetTransformers, create_pretrained_resnet50


def run(model_dir: Path, params):
    model_path = str(model_dir.joinpath("model"))
    data_folder = BoxDataFolder(transform=ImagenetTransformers())
    batch_size = 64
    save_dir = model_dir.joinpath("evaluation")
    save_dir.mkdir(parents=True, exist_ok=True)

    to_indices = lambda file: [int(idx) for idx in model_dir.joinpath(file).read_text().splitlines(keepends=False)]
    ModelEvaluation(create_pretrained_resnet50, params, model_path, save_dir).evaluate_dataset(
        data_folder,
        train_indices=to_indices("train_indices.csv"),
        valid_indices=to_indices("valid_indices.csv"),
        test_indice=to_indices("test_indices.csv"),
        batch_size=batch_size
    )
