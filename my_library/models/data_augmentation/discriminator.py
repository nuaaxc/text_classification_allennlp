from typing import Dict

import torch
import torch.nn as nn

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules import FeedForward
from allennlp.models.model import Model


@Model.register("discriminator-base")
class Discriminator(Model):
    """
    A model that takes a sample (input_dim,) and tries to predict 1
    if it's from the true distribution and 0 if it's from the generator.
    """
    def __init__(self,
                 label_emb: TokenEmbedder,
                 feed_forward: FeedForward) -> None:
        super().__init__(None)
        self.label_emb = label_emb
        self.feed_forward = feed_forward
        self.loss = torch.nn.BCELoss()

    def forward(self,  # type: ignore
                text: torch.Tensor,
                label: torch.Tensor,
                validity_labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        embbed_label = self.label_emb(label)
        output = torch.sigmoid(self.feed_forward(torch.cat([text, embbed_label], dim=-1)))
        output_dict = {"output": output}
        if validity_labels is not None:
            output_dict["loss"] = self.loss(output, validity_labels)
        return output_dict
