import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import resample
from torch.utils.data import DataLoader as TorchDataLoader, SubsetRandomSampler


class BoxDataFolder(torchvision.datasets.ImageFolder):
    DEFAULT_BOX_IMAGES_PATH = Path(__file__).parent.joinpath("box-images")

    def __init__(self, transform=None, target_transform=None):
        super().__init__(str(self.DEFAULT_BOX_IMAGES_PATH), transform, target_transform)

    @staticmethod
    def create_train_test_loader(test_size, train_batch_size, test_batch_size, train_transform=None,
                                 test_transform=None,
                                 random_seed=None, num_workers=None):
        train_dataset = BoxDataFolder(train_transform)
        test_dataset = BoxDataFolder(test_transform)

        if num_workers is None:
            num_workers = os.cpu_count()

        n_samples = len(train_dataset)
        targets = train_dataset.targets

        train_indices, test_indices = train_test_split(range(n_samples), shuffle=True, test_size=test_size,
                                                       random_state=random_seed,
                                                       stratify=targets)
        train_loader = TorchDataLoader(train_dataset, batch_size=train_batch_size,
                                       sampler=SubsetRandomSampler(train_indices), num_workers=num_workers)
        test_loader = TorchDataLoader(test_dataset, batch_size=test_batch_size,
                                      sampler=SubsetRandomSampler(test_indices), num_workers=num_workers)
        return train_loader, test_loader


def sample_indices(indices, aug_ratio, random_seed=None):
    samples = indices
    if aug_ratio > 0:
        samples = np.hstack([samples, resample(samples, replace=True, n_samples=int(len(samples) * aug_ratio),
                                               random_state=random_seed)])
    return samples


def to_kfold_dataloader(dataloader: TorchDataLoader, nfold, valid_transform=None, random_seed=None, std_scale=False,
                        train_batch_size=None, valid_batch_size=None, aug_ratio=0, num_workers=None):
    train_indices = dataloader.sampler.indices
    train_targets = [dataloader.dataset.targets[idx] for idx in train_indices]

    kfolds = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=random_seed).split(
        train_indices, train_targets)
    valid_data_folder = BoxDataFolder(valid_transform)

    for idx, (train_indices, valid_indices) in enumerate(kfolds):
        # if std_scale:
        #     train_images = torch.stack([dataloader.dataset[i][0] for i in train_indices], dim=0)
        #     train_mean = train_images.mean(dim=0)
        #     train_std = train_images.std(dim=0)
        #     del train_images
        #     gc.collect()
        #     if idx == 0:
        #         dataloader.dataset.transform.transform.insert(0,
        #                                                       torchvision.transforms.Normalize(mean=train_mean,
        #                                                                                        std=train_std))
        #         valid_data_folder.transform.transform.insert(0,
        #                                                      torchvision.transforms.Normalize(mean=train_mean,
        #                                                                                       std=train_std))
        #     else:
        #         dataloader.dataset.transform.transform[0] = torchvision.transforms.Normalize(mean=train_mean,
        #                                                                                      std=train_std)
        #         valid_data_folder.transform.transform[0] = torchvision.transforms.Normalize(mean=train_mean,
        #                                                                                     std=train_std)
        train_batch_size = train_batch_size if not train_batch_size else dataloader.batch_size
        valid_batch_size = valid_batch_size if not valid_batch_size else dataloader.batch_size
        num_workers = num_workers if not num_workers else dataloader.num_workers

        train_indices = sample_indices(train_indices, aug_ratio, random_seed=random_seed)

        yield (TorchDataLoader(dataloader.dataset, batch_size=train_batch_size,
                               sampler=SubsetRandomSampler(train_indices), num_workers=num_workers),
               TorchDataLoader(valid_data_folder, batch_size=dataloader.batch_size,
                               sampler=SubsetRandomSampler(valid_indices), num_workers=num_workers))


def to_holdout_dataloader(dataloader: TorchDataLoader, valid_size, valid_transform=None, random_seed=None,
                          std_scale=False,
                          train_batch_size=None, valid_batch_size=None, aug_ratio=0, num_workers=None):
    train_indices = dataloader.sampler.indices
    train_targets = [dataloader.dataset.targets[idx] for idx in train_indices]

    train_indices, valid_indices = train_test_split(train_indices, shuffle=True, test_size=valid_size,
                                                    random_state=random_seed,
                                                    stratify=train_targets)
    augmented_train_indices = train_indices
    if aug_ratio > 0:
        augmented_train_indices = sample_indices(train_indices, aug_ratio, random_seed=random_seed)

    pin_memory = torch.cuda.is_available()
    train_data_loader = TorchDataLoader(dataloader.dataset, batch_size=train_batch_size,
                                        sampler=SubsetRandomSampler(augmented_train_indices), num_workers=num_workers,
                                        pin_memory=pin_memory)

    valid_data_folder = BoxDataFolder(valid_transform)
    valid_data_loader = TorchDataLoader(valid_data_folder, batch_size=valid_batch_size,
                                        sampler=SubsetRandomSampler(valid_indices), num_workers=num_workers,
                                        pin_memory=pin_memory)

    return train_data_loader, valid_data_loader, train_indices, valid_indices


if __name__ == '__main__':
    # for debug
    train_loader, test_loader = BoxDataFolder.create_train_test_loader(test_size=0.2, train_batch_size=32,
                                                                       test_batch_size=32,
                                                                       train_transform=torchvision.transforms.Compose(
                                                                           [torchvision.transforms.ToTensor()]
                                                                       ),
                                                                       test_transform=torchvision.transforms.Compose(
                                                                           [torchvision.transforms.ToTensor()]
                                                                       )
                                                                       )
    for train, valid in to_kfold_dataloader(train_loader, 5, std_scale=True,
                                            valid_transform=torchvision.transforms.Compose(
                                                [torchvision.transforms.ToTensor()]
                                            )):
        print(train)
        print(valid)
