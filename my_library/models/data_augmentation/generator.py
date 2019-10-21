from typing import Dict
import torch
# import torch.nn as nn
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules import FeedForward
from allennlp.models.model import Model


@Model.register('generator-base')
class Generator(Model):

    def __init__(self,
                 label_emb: TokenEmbedder,
                 perturb: FeedForward,
                 feed_forward: FeedForward) -> None:
        super().__init__(None)
        self.label_emb = label_emb
        self.perturb = perturb
        self.feed_forward = feed_forward

    def forward(self,
                feature: torch.Tensor,
                noise: torch.Tensor,
                label: torch.Tensor,
                discriminator: Model = None) -> Dict[str, torch.Tensor]:
        perturbation = self.perturb(torch.cat([feature, noise], dim=-1))
        label = label.long()
        emb_label = self.label_emb(label)
        features = self.feed_forward(torch.cat([perturbation, emb_label], dim=-1))
        output_dict = {'output': features}
        if discriminator is not None:
            fake_validity = discriminator(features, label)['output']
            output_dict['loss'] = -torch.mean(fake_validity)
        return output_dict



