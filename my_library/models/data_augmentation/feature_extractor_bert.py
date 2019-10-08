from typing import Dict, Union, Optional
import torch

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.seq2vec_encoders import BertPooler
from allennlp.nn.util import get_text_field_mask

from my_library.models.layers import bert_embeddings


@Model.register("bert_encoder")
class BertEncoder(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            bert_path: str,
            dropout: float = 0.0,
            trainable: bool = False,
    ) -> None:
        super().__init__(vocab)

        self._embeddings = bert_embeddings(pretrained_model=bert_path,
                                           training=trainable)
        self.pooler = BertPooler(pretrained_model=bert_path,
                                 dropout=dropout)

    def forward(  # type: ignore
            self, text: Dict[str, torch.LongTensor]) -> torch.Tensor:
        mask = get_text_field_mask(text)
        emb = self._embeddings(text)
        encoding = self.pooler(emb, mask)

        return encoding
