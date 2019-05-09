from pathlib import Path

from box_dataset import BoxDataFolder
from model_wrapper import NnModelTrainer
from models import create_pretrained_resnet50, ImagenetTransformers, ImagenetAugmentTransformers
from scripts.common_cfg import RANDOM_SEED


def main():
    params = {
        "dropout_rate": 0.4,
        "inner_units": 16
    }

    src_dataset = BoxDataFolder(transform=ImagenetTransformers())
    train_dataloader, test_dataloader = src_dataset.create_train_test_loader(
        test_size=0.2,
        train_batch_size=32, test_batch_size=32,
        random_seed=RANDOM_SEED,
        train_transform=ImagenetAugmentTransformers(),
        test_transform=ImagenetTransformers()
    )

    save_dir = Path(__file__).parent.parent.joinpath("output/resnet50_pretrained")
    save_dir.mkdir(exist_ok=True, parents=True)

    NnModelTrainer.write_indices(test_dataloader.sampler.indices, "test", save_dir)
    model = NnModelTrainer(params, model_factory=create_pretrained_resnet50, save_dir=save_dir, lr=1e-2,
                           clip_grad_value=10, random_state=RANDOM_SEED)
    score_avg, score_mean, _, _ = model.kfold_train(data_loader=train_dataloader,
                                                    train_batch_size=32,
                                                    valid_batch_size=64,
                                                    n_epochs=400,
                                                    patience=50,
                                                    aug_ratio=2,
                                                    random_seed=RANDOM_SEED)

    print("score avg: {:.6f} std: {:.6f}".format(score_avg, score_mean))
    print("done")


if __name__ == '__main__':
    main()
