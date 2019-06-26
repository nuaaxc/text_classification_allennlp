from typing import Dict

import torch
import torch.nn as nn

from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure


@Model.register("classifier-base")
class Classifier(Model):
    """
    A model that takes a sample (input_dim,) and tries to predict 1
    if it's from the true distribution and 0 if it's from the generator.
    """
    def __init__(self,
                 feed_forward: FeedForward) -> None:
        super().__init__(None)
        self.feed_forward = feed_forward
        self.loss = torch.nn.CrossEntropyLoss()
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "f1": FBetaMeasure()
        }

    def forward(self,  # type: ignore
                inputs: torch.Tensor,
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        output = self.feed_forward(inputs)
        output = torch.softmax(output, dim=-1)
        output_dict = {"output": output}
        if labels is not None:
            output_dict["loss"] = self.loss(output, labels)
            for metric in self.metrics.values():
                metric(output, labels.squeeze(-1))

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset)
                for metric_name, metric in self.metrics.items()}
