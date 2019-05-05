from typing import Dict, Optional, List

import numpy
from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder
from allennlp.modules.seq2vec_encoders import  BagOfEmbeddingsEncoder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy

from my_library.loss.metric_loss import TripletLoss


@Model.register("base_classifier")
class BaseLSTM(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 d_hidden: int,
                 dropout: float,
                 cuda_device: int,
                 _lambda: float,
                 ) -> None:
        super(BaseLSTM, self).__init__(vocab)

        self.cuda_device = cuda_device
        self._lambda = _lambda
        self.word_embeddings = word_embeddings
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.encoder = encoder
        self.feature_layer = FeedForward(input_dim=self.encoder.get_output_dim(),
                                         num_layers=4,
                                         hidden_dims=d_hidden,
                                         activations=F.relu,
                                         dropout=dropout)
        self.classifier = nn.Linear(d_hidden, self.num_classes)
        self.dropout = nn.Dropout(dropout)
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
        }
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self,  # type: ignore
                epoch_num: List[int],
                tokens: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        mask = get_text_field_mask(tokens)
        tokens = self.word_embeddings(tokens)
        tokens = self.encoder(tokens, mask)
        tokens = self.feature_layer(tokens)
        logits = self.classifier(tokens)

        output_dict = {'logits': logits}

        if labels is not None:
            class_loss = self.loss_func(logits, labels)
            online_triplet_loss = TripletLoss(labels, tokens, margin=0.5,
                                              squared=False, cuda_device=self.cuda_device)
            triplet_loss = online_triplet_loss.batch_hard_triplet_loss()
            for metric in self.metrics.values():
                metric(logits, labels)
            # print(triplet_loss)
            output_dict['loss'] = class_loss + self._lambda * triplet_loss

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
