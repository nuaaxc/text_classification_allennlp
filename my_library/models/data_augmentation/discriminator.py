from typing import Dict

import torch
import torch.nn as nn

from allennlp.common.checks import ConfigurationError
from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.nn.activations import Activation

from my_library.models.utils import get_moments


@Model.register("discriminator-base")
class Discriminator(Model):
    """
    A model that takes a sample (input_dim,) and tries to predict 1
    if it's from the true distribution and 0 if it's from the generator.
    """

    def __init__(self,
                 d_input: int,
                 d_hidden: int,
                 n_layers: int,
                 dropout: float,
                 activation: Activation = torch.nn.Sigmoid(),
                 preprocessing: str = None) -> None:
        super().__init__(None)
        assert n_layers >= 2
        if preprocessing is None:
            self.preprocess = lambda x: x
        elif preprocessing == "moments":
            self.preprocess = get_moments
            d_input = 4
        else:
            raise ConfigurationError("unknown preprocessing")

        self.model = FeedForward(input_dim=d_input,
                                 num_layers=n_layers,
                                 hidden_dims=[d_hidden] * (n_layers - 1) + [1],
                                 activations=activation,
                                 dropout=dropout)

        self.loss = torch.nn.BCELoss()

    def forward(self,  # type: ignore
                inputs: torch.Tensor,
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        inputs = inputs.squeeze(-1)
        output = self.model(self.preprocess(inputs))
        output_dict = {"output": output}
        if label is not None:
            output_dict["loss"] = self.loss(output, label)

        return output_dict
