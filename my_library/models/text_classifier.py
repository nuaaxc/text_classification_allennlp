from typing import Dict

import numpy as np

import torch
import torch.nn as nn

from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.data import Vocabulary
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure
from allennlp.nn.util import get_text_field_mask


@Model.register("text-classifier-base")
class TextClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 feed_forward: FeedForward,
                 dropout: float = None,
                 initializer: InitializerApplicator = InitializerApplicator()
                 ) -> None:
        super().__init__(vocab)
        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.feed_forward = feed_forward
        self.dropout = torch.nn.Dropout(dropout)
        self.loss = torch.nn.CrossEntropyLoss()
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
        }
        initializer(self)

    def forward(  # type: ignore
            self,
            tokens,
            labels: torch.IntTensor = None,
    ) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        embeddings = self.text_field_embedder(tokens)
        encoder_out = self.encoder(embeddings, mask)
        encoder_out = self.dropout(encoder_out)
        logits  = self.feed_forward(encoder_out)
        probs = torch.softmax(logits, dim=-1)
        output_dict = {'probs': probs, 'logits': logits}

        if labels is not None:
            labels = labels.long().view(-1)
            output_dict["loss"] = self.loss(logits, labels)
            for metric in self.metrics.values():
                metric(probs, labels)

        return output_dict
