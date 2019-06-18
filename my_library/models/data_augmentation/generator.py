from typing import Dict

import torch
import torch.nn as nn

from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.nn.activations import Activation


@Model.register('generator-base')
class Generator(Model):

    def __init__(self,
                 d_input: int,
                 d_hidden: int,
                 n_layers: int,
                 dropout: float,
                 activation: Activation = nn.ReLU()) -> None:
        super(Generator, self).__init__(None)

        self.model = FeedForward(input_dim=d_input,
                                 num_layers=n_layers,
                                 hidden_dims=d_hidden,
                                 activations=activation,
                                 dropout=dropout)
        self.loss = nn.BCELoss()

    def forward(self,
                z: torch.Tensor,
                discriminator: Model = None) -> Dict[str, torch.Tensor]:
        output = self.model(z)
        output_dict = {'output': output}
        if discriminator is not None:
            predicted = discriminator(output)['output']
            desired = torch.ones_like(predicted)
            output_dict['loss'] = self.loss(predicted, desired)
        return output_dict



