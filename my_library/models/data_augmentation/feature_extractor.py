from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn.util import get_text_field_mask


@Model.register("feature-extractor-base")
class FeatureExtractor(Model):

    def __init__(self,
                 text_field_embedder: TextFieldEmbedder,
                 text_encoder: Seq2VecEncoder,
                 d_hidden: int,
                 dropout: float,
                 ) -> None:
        super(FeatureExtractor, self).__init__(None)

        self.text_field_embedder = text_field_embedder
        self.text_encoder = text_encoder
        self.dense = FeedForward(input_dim=self.text_encoder.get_output_dim(),
                                 num_layers=2,
                                 hidden_dims=d_hidden,
                                 activations=F.relu,
                                 dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        tokens = self.text_field_embedder(tokens)
        tokens = self.text_encoder(tokens, mask)
        output = self.dense(tokens)
        output_dict = {"output": output}

        return output_dict

