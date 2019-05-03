from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("base_classifier")
class BaseLSTM(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 n_label: int,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder) -> None:
        super(BaseLSTM, self).__init__(vocab)

        self.word_embeddings = word_embeddings
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.encoder = encoder
        self.classifier = nn.Linear(self.encoder.get_output_dim(), n_label)
        self.metrics = {
                "accuracy": CategoricalAccuracy(),
        }
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        mask = get_text_field_mask(tokens)
        tokens = self.word_embeddings(tokens)

        encoded_text = self.encoder(tokens, mask)
        logits = self.classifier(encoded_text)
        output_dict = {'logits': logits}

        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
