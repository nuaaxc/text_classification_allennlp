from typing import Dict

import torch
import torch.nn as nn

from allennlp.modules import FeedForward
from allennlp.models.model import Model


@Model.register('generator-base')
class Generator(Model):

    def __init__(self,
                 feed_forward: FeedForward) -> None:
        super(Generator, self).__init__(None)
        self.feed_forward = feed_forward
        self.loss = nn.BCELoss()

    def forward(self,
                z: torch.Tensor,
                discriminator: Model = None) -> Dict[str, torch.Tensor]:
        output = self.feed_forward(z)
        output_dict = {'output': output}
        if discriminator is not None:
            predicted = discriminator(output)['output']
            desired = torch.ones_like(predicted)
            output_dict['loss'] = self.loss(predicted, desired)
        return output_dict



