from unittest import TestCase

import torch

from model_wrapper import CrossEntropyCosineLoss, CosineLoss


class TestCrossEntropyCosineLoss(TestCase):

    def test(self):
        loss_function = CrossEntropyCosineLoss(lambda_=0.3)
        inputs_ = torch.Tensor([[0.99, 0.89], [0.55, 0.33]])
        labels_ = torch.Tensor([[0., 1.], [1., 0.]])
        output_ = loss_function(inputs_, labels_)
        print(output_)
        # TODO assertion


class TestCosineLoss(TestCase):
    def test(self):
        loss_function = CosineLoss()
        inputs_ = torch.Tensor([[0.99, 0.01], [0.55, 0.33]])
        labels_ = torch.Tensor([[0., 1.], [1., 0.]])
        output_ = loss_function(inputs_, labels_)
        print(output_)
        # TODO assertion
