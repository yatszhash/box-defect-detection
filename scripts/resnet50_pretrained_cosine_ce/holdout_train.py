from pathlib import Path

from box_dataset import BoxDataFolder
from model_wrapper import ImagenetAugmentTransformers, ImagenetTransformers, NnModelWrapper
from models import PretrainedResnet50WithClassEmbedding
from scripts.common_cfg import RANDOM_SEED


def main():
    params = {
        "dropout_rate": 0.7,
        "inner_units": 16,
        "class_embedding": 2,
        "n_class": 1
    }

    src_dataset = BoxDataFolder(transform=ImagenetTransformers())
    train_dataloader, test_dataloader = src_dataset.create_train_test_loader(
        test_size=0.2,
        train_batch_size=32, test_batch_size=32,
        random_seed=RANDOM_SEED,
        train_transform=ImagenetAugmentTransformers(),
        test_transform=ImagenetTransformers()
    )

    save_dir = Path(__file__).parent.parent.parent.joinpath("output/resnet50_pretrained_cosine_ce/holdout")
    save_dir.mkdir(exist_ok=True, parents=True)

    NnModelWrapper.write_indices(test_dataloader.sampler.indices, "test", save_dir)
    model = NnModelWrapper(params, model_factory=PretrainedResnet50WithClassEmbedding, save_dir=save_dir, lr=1e-1,
                           clip_grad_value=10, random_state=RANDOM_SEED, loss_function="cosine_cross_entropy",
                           loss_lambda_=0.1)
    score, result_df = model.holdout_train(data_loader=train_dataloader,
                                           train_batch_size=64,
                                           valid_batch_size=128,
                                           n_epochs=400,
                                           patience=100,
                                           valid_size=0.2,
                                           aug_ratio=2,
                                           random_seed=RANDOM_SEED)

    print("best rvalidation score: {:.6f}".format(score))
    print("done")


if __name__ == '__main__':
    main()
