from typing import Dict, Any

import torch

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn.util import get_text_field_mask


@Model.register("feature-extractor-base")
class FeatureExtractor(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 text_encoder: Seq2VecEncoder,
                 feed_forward: FeedForward,
                 ) -> None:
        super().__init__(vocab)

        self.text_field_embedder = text_field_embedder
        self.text_encoder = text_encoder
        self.feed_forward = feed_forward

    def forward(self, text: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(text)
        embeddig = self.text_field_embedder(text)
        encoding = self.text_encoder(embeddig, mask)
        output = self.feed_forward(encoding)
        return output

