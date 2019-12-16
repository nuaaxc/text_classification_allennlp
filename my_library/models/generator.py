from typing import Dict
import torch
# import torch.nn as nn
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules import FeedForward
from allennlp.models.model import Model


@Model.register('generator-base')
class Generator(Model):

    def __init__(self,
                 perturb: FeedForward,
                 feed_forward: FeedForward,
                 label_emb: TokenEmbedder) -> None:
        super().__init__(None)
        self.label_emb = label_emb
        self.feed_forward = feed_forward
        self.perturb = perturb

    def forward(self,
                feature: torch.Tensor,
                noise: torch.Tensor,
                label: torch.Tensor,
                discriminator: Model = None) -> Dict[str, torch.Tensor]:
        emb_label = self.label_emb(label.long())
        feature_aug = self.perturb(torch.cat([feature, noise], dim=-1))
        feature_aug = self.feed_forward(torch.cat([feature_aug, emb_label], dim=-1))

        output_dict = {'output': feature_aug}

        if discriminator is not None:
            fake_validity = discriminator(feature_aug, label)['output']
            output_dict['loss'] = -torch.mean(fake_validity)
        return output_dict
