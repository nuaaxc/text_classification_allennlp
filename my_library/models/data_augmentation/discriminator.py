from typing import Dict

import torch
import torch.nn as nn

from allennlp.modules import FeedForward
from allennlp.models.model import Model


@Model.register("discriminator-base")
class Discriminator(Model):
    """
    A model that takes a sample (input_dim,) and tries to predict 1
    if it's from the true distribution and 0 if it's from the generator.
    """
    def __init__(self,
                 feed_forward: FeedForward) -> None:
        super().__init__(None)
        self.feed_forward = feed_forward
        self.loss = torch.nn.BCELoss()

    def forward(self,  # type: ignore
                inputs: torch.Tensor,
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        output = self.feed_forward(inputs)
        # output = torch.clamp(output, min=0., max=1.)
        output = torch.sigmoid(output)
        output_dict = {"output": output}
        if labels is not None:
            output_dict["loss"] = self.loss(output, labels)

        return output_dict
