import logging
import os
import sys
from abc import ABCMeta
from pathlib import Path

import numpy as np
import pandas as pd
import tensorboardX as tbx
import torch
import torch.nn.functional as F
import torchvision
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import binarize
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from box_dataset import to_kfold_dataloader, to_holdout_dataloader
from models import PretrainedResnet50WithClassEmbedding, ImagenetTransformers, CosineLoss, CosineCrossEntropyLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
logger.addHandler(stdout_handler)


class NnModelTrainer(object, metaclass=ABCMeta):
    '''
    ported from my own kaggle repository
    '''

    def __init__(self, model_params, model_factory, save_dir: Path, optimizer_factory=None, loss_function=None,
                 score_function=None, lr=1e-3, weight_decay=1e-4, scheduler="cosine_annealing", clip_grad_value=None,
                 threshold=0.5, model_path=None,
                 random_state=0, prepruning=None, **kwargs):
        self.random_state = random_state
        self.threshold = threshold
        self.model_path = model_path
        self.clip_grad_value = clip_grad_value
        self.scheduler = scheduler
        self.weight_decay = weight_decay
        self.lr = lr
        self.loss_function = loss_function
        self.optimizer_factory = optimizer_factory
        self.model_factory = model_factory
        self.model_params = model_params

        self.save_root = save_dir
        self._current_save_dir = save_dir
        self.prepruning = prepruning
        self.kwargs = kwargs

        if score_function:
            self.score_function = score_function
        else:
            self.score_function = lambda y_true, y_pred: f1_score(y_true.reshape((-1)),
                                                                  binarize(y_pred,
                                                                           threshold=threshold).reshape((-1)))

        # self.scheduler = StepLR(self.optimizer, step_size=20)

    def _create_new_model(self):
        self.model = self.model_factory(**self.model_params)

        if not self.clip_grad_value:
            for p in self.model.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, min=-self.clip_grad_value, max=self.clip_grad_value))

        if self.prepruning is not None:
            for name, threshold in self.prepruning:
                mask = self.model.named_parameters()[name].data.abs() < threshold
                layer = self.model.named_parameters()[name]
                layer.data = layer.data.masked_fill_(mask, 0.0)
                layer.register_hook(lambda grad: grad.masked_fill_(mask, threshold))

        self._model_to_device(self.random_state)

        if self.model_path:
            self.model.load_state_dict(torch.load(str(self.model_path)))

        self.n_epoch = None
        self._current_epoch = 0
        self._current_max_valid_score = 0
        self._early_stop_count = 0

        self._current_model_save_path = self._current_save_dir.joinpath("model")

        self.train_result_path = self._current_save_dir.joinpath("resul t.csv")
        self.train_results = pd.DataFrame()

        self.register_optimizer(self.lr, self.optimizer_factory, self.weight_decay)
        self._register_scheduler(self.scheduler)
        self.register_loss_function(self.loss_function)

    def _register_scheduler(self, scheduler):
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=12,
                                           eta_min=1e-6) if scheduler == "cosine_annealing" else scheduler(
            self.optimizer)

    def _model_to_device(self, random_state):
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
            if torch.cuda.device_count() >= 2:
                self.model = torch.nn.DataParallel(self.model).cuda()
            else:
                self.model = self.model.cuda()

    def register_optimizer(self, lr, optimizer_factory, weight_decay):
        if not optimizer_factory:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optimizer_factory(self.model.parameters(), lr)

    def register_loss_function(self, loss_function):
        self.require_softmax = True
        if loss_function == "cosine":
            self.loss_function = CosineLoss()
        elif loss_function == "cosine_cross_entropy":
            self.loss_function = CosineCrossEntropyLoss(self.kwargs["loss_lambda_"])
            self.require_softmax = False
        elif loss_function == "mae":
            self.loss_function = nn.L1Loss()
        else:
            self.loss_function = nn.BCEWithLogitsLoss()
            self.require_softmax = False

    def kfold_train(self, data_loader: DataLoader, train_batch_size, valid_batch_size, n_epochs, patience=10,
                    validation_metric="score",
                    nfold=5, num_workers=None, random_seed=None, aug_ratio=0):

        if num_workers is None:
            num_workers = os.cpu_count()

        kfolds = to_kfold_dataloader(data_loader, valid_transform=ImagenetTransformers(), nfold=nfold,
                                     train_batch_size=train_batch_size,
                                     valid_batch_size=valid_batch_size, random_seed=random_seed,
                                     num_workers=num_workers, aug_ratio=aug_ratio)
        kfold_result_dfs = []
        kfold_validation_scores = []
        for fold_idx, (train_data_loader, valid_data_loader) in enumerate(kfolds):
            logger.info("############ n_fold {} ##########".format(fold_idx))
            self._current_save_dir = self.save_root.joinpath(str(fold_idx))
            val_score, result_df = self.train(train_data_loader, valid_data_loader, n_epochs, patience=patience,
                                              validation_metric=validation_metric)
            kfold_result_dfs.append(result_df)
            kfold_validation_scores.append(kfold_validation_scores)

        return np.mean(kfold_validation_scores), np.std(kfold_validation_scores), \
               kfold_result_dfs, kfold_validation_scores

    def holdout_train(self, data_loader: DataLoader, train_batch_size, valid_batch_size, n_epochs, patience=10,
                      validation_metric="score", valid_size=0.2,
                      num_workers=None, random_seed=None, aug_ratio=0):

        if num_workers is None:
            num_workers = os.cpu_count()

        train_data_loader, valid_data_loader, train_indices, valid_indices = to_holdout_dataloader(
            data_loader,
            valid_transform=ImagenetTransformers(),
            valid_size=valid_size,
            train_batch_size=train_batch_size,
            valid_batch_size=valid_batch_size,
            random_seed=random_seed,
            num_workers=num_workers,
            aug_ratio=aug_ratio)

        self.write_indices(train_indices, "train", self._current_save_dir)
        self.write_indices(valid_indices, "valid", self._current_save_dir)

        val_score, result_df = self.train(train_data_loader, valid_data_loader, n_epochs, patience=patience,
                                          validation_metric=validation_metric)

        result_df.to_csv(self._current_save_dir.joinpath("train_result.csv"))
        return val_score, result_df

    @staticmethod
    def write_indices(indices, suffix, save_dir):
        with save_dir.joinpath(suffix + "_indices.csv").open(mode="w+", encoding="utf-8") as f:
            f.write("\n".join([str(idx) for idx in indices]))

    def train(self, train_data_loader: DataLoader, valid_data_loader: DataLoader, n_epochs, patience=10,
              validation_metric="score"):
        self._create_new_model()
        tensorbord_log = self._current_save_dir.joinpath("tesnsorbord_log")
        tensorbord_log.mkdir(parents=True, exist_ok=True)
        self._tbx_writer = tbx.SummaryWriter(str(tensorbord_log))
        self.clear_history()
        if validation_metric != "score":
            self._current_max_valid_score = - np.inf

        self.patience = patience
        self._train_dataloader = train_data_loader
        self._valid_dataloader = valid_data_loader
        self.n_epoch = n_epochs

        logger.info("train with data size: {}".format(len(self._train_dataloader.sampler)))
        logger.info("valid with data size: {}".format(len(self._valid_dataloader.sampler)))

        iterator = tqdm(range(n_epochs))
        for epoch in iterator:
            self._current_epoch = epoch + 1
            logger.info("training %d  / %d epochs", self._current_epoch, n_epochs)
            # self.scheduler.step()
            self._train_epoch(epoch)
            self.write_current_result()
            self._valid_epoch(epoch)
            self.write_current_result()

            if validation_metric == "score":
                valid_metric_value = self.train_results["valid_score"][self._current_epoch]
            else:
                valid_metric_value = - self.train_results["valid_loss"][self._current_epoch]

            if valid_metric_value <= self._current_max_valid_score:
                self._early_stop_count += 1
                logger.info("validation metric isn't improved")
            else:
                logger.info("validation metric is improved from %.5f to %.5f",
                            self._current_max_valid_score, valid_metric_value)
                self._current_max_valid_score = valid_metric_value
                self._early_stop_count = 0
                self.save_models()

            if self._early_stop_count >= self.patience:
                logger.info("======early stopped=====")
                self.model.load_state_dict(torch.load(self._current_model_save_path))
                iterator.close()
                break

            self.scheduler.step()

        logger.info("train done! best validation metric : %.6f", self._current_max_valid_score)

        self._tbx_writer.export_scalars_to_json(self._current_save_dir.joinpath("all_scalars.json"))
        self._tbx_writer.close()
        return self._current_max_valid_score, self.train_results

    def write_current_result(self):
        self.train_results.to_csv(self.train_result_path, encoding="utf-8")

    def clear_history(self):
        self.n_epoch = None
        self._current_epoch = 0
        self.train_results = pd.DataFrame()

        self._current_max_valid_score = 0
        self._early_stop_count = 0

    def _train_epoch(self, epoch):
        self.model.train()

        all_labels = []
        all_outputs = []
        total_loss = 0.0
        for i, data in enumerate(self._train_dataloader):
            # print("batch data size {}".format(inputs.size()))
            inputs, labels, device = self.switch_device(data)

            self.optimizer.zero_grad()

            loss = self.compute_batch_loss(all_labels, all_outputs, labels, inputs)
            loss.backward()
            self.optimizer.step()
            # self.optimizer.zero_grad()
            total_loss += loss.cpu().detach().item()
            if i % 2000 == 1999:
                logger.info('[%d, %5d] loss: %.7f' %
                            (self._current_epoch, i + 1, total_loss / (i + 1)))
        self.optimizer.zero_grad()

        avg_loss = total_loss / len(self._train_dataloader)
        logger.info("******train loss at epoch %d: %.7f :" % (self._current_epoch, avg_loss))
        self.train_results.loc[self._current_epoch, "train_loss"] = avg_loss
        self._tbx_writer.add_scalar('loss/train_loss', avg_loss, epoch)
        all_outputs = np.vstack(all_outputs)
        all_labels = np.vstack(all_labels)
        score = self.score_function(all_labels, all_outputs)
        logger.info("******train score at epoch %d: %.5f :" % (self._current_epoch, score))
        self._tbx_writer.add_scalar('score/train_score', score, epoch)
        self.train_results.loc[self._current_epoch, "train_score"] = score

    def compute_batch_loss(self, all_labels, all_outputs, labels, inputs):
        outputs = self.model(inputs)

        if isinstance(self.model, PretrainedResnet50WithClassEmbedding):
            class_embedding, outputs = outputs[0], outputs[1]
        labels = labels.reshape((-1, 1)).float()
        all_labels.append(labels.cpu().detach().numpy())
        if not self.require_softmax:
            predicted_classes = torch.sigmoid(outputs)
        else:
            predicted_classes = torch.softmax(outputs, dim=1)[:, 1:]
        all_outputs.append(predicted_classes.cpu().detach().numpy())
        # labels = labels.to(device)
        if isinstance(self.model, PretrainedResnet50WithClassEmbedding):
            return self.loss_function(outputs, class_embedding, labels)
        return self.loss_function(outputs, labels)

    def switch_device(self, data):
        inputs = data[0]
        labels = data[1]
        if torch.cuda.is_available() and not inputs.is_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return inputs, labels, device

    def _valid_epoch(self, epoch):
        total_loss = 0.0

        all_labels = []
        all_outputs = []
        self.model.eval()
        for i, data in enumerate(self._valid_dataloader):

            inputs, labels, device = self.switch_device(data)

            loss = self.compute_batch_loss(all_labels, all_outputs, labels, inputs)

            total_loss += loss.cpu().detach().item()
            if i % 2000 == 1999:
                logger.info('[%d, %5d] validation loss: %.7f' %
                            (self._current_epoch, i + 1, total_loss / (i + 1)))

        avg_loss = total_loss / len(self._valid_dataloader)
        logger.info("******valid loss at epoch %d: %.7f :" % (self._current_epoch, avg_loss))
        self.train_results.loc[self._current_epoch, "valid_loss"] = avg_loss
        self._tbx_writer.add_scalar('loss/valid_loss', avg_loss, epoch)
        all_outputs = np.vstack(all_outputs)
        all_labels = np.vstack(all_labels)
        score = self.score_function(all_labels, all_outputs)
        self._tbx_writer.add_scalar('score/valid_score', score, epoch)
        logger.info("******valid score at epoch %d: %.5f :" % (self._current_epoch, score))

        self.train_results.loc[self._current_epoch, "valid_score"] = score

    def save_models(self):
        torch.save(self.model.state_dict(), str(self._current_model_save_path))
        logger.info("Checkpoint saved")

    def predict(self, dataloader: DataLoader, batch_size, n_job):
        logger.info("predicting {} samples...".format(len(dataloader.dataset)))

        self.model.eval()
        sigmoid = nn.Sigmoid()
        return np.vstack([sigmoid(self.model(x[0])).cpu().detach().numpy() for x in tqdm(dataloader)])

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight.data)
            m.bias.data.zero_()


class ModelEvaluation(object):

    def __init__(self, model_factory, model_params, model_path, save_dir):
        self.model = model_factory(**model_params).cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.save_dir = save_dir

    def evaluate_dataset(self, data_folder: torchvision.datasets.ImageFolder, train_indices, valid_indices, test_indice,
                         batch_size, num_workers=None):
        self._datafolder = data_folder
        self.scores_df = pd.DataFrame()
        num_workers = os.cpu_count() if num_workers is None else num_workers

        self._evaluate(train_indices, batch_size, "train", num_workers)
        self._evaluate(valid_indices, batch_size, "valid", num_workers)
        self._evaluate(test_indice, batch_size, "test", num_workers)

        self.scores_df.to_csv(self.save_dir.joinpath("evaluation_score.csv"))
        print(self.scores_df)

    def _evaluate(self, indices, batch_size, subset_name, num_workers=None):
        subset = Subset(self._datafolder, indices)

        logger.info("predicting {} samples...".format(len(subset)))

        self.model.eval()
        predicted_df = pd.DataFrame()
        predicted_df["file_name"] = [self._datafolder.samples[idx][0] for idx in indices]
        predicted_df["target"] = [self._datafolder.targets[idx] for idx in indices]

        predicted_df["predicted"] = np.vstack([self._predict(x).cpu().detach().numpy() for x
                                               in tqdm(
                DataLoader(dataset=subset, shuffle=False, batch_size=batch_size, num_workers=num_workers))])
        predicted_df["class"] = binarize(predicted_df["predicted"].values.reshape((-1, 1)), threshold=0.5)

        predicted_df.to_csv(self.save_dir.joinpath(f'{subset_name}_predicted.csv'))
        self.scores_df = self.scores_df.append({
            "subset": subset_name,
            "n_samples": len(subset),
            "f1_scores": f1_score(predicted_df["target"].values, predicted_df["class"].values),
            "roc_auc": roc_auc_score(predicted_df["target"].values, predicted_df["predicted"].values)
        }, ignore_index=True)

    def _predict(self, x):
        output = self.model(x[0].cuda())
        if output.shape[1] == 2:
            return torch.softmax(output, dim=1)[:, 1:]
        return torch.sigmoid(output)
