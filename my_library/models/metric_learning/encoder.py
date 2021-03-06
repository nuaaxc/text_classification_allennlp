import torch
from overrides import overrides
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder


class BertSentencePooling(Seq2VecEncoder):
    def __init__(self, vocab, bert_dim):
        super(BertSentencePooling, self).__init__(vocab)
        self.bert_dim = bert_dim

    @overrides
    def forward(self,
                emb: torch.tensor,
                mask: torch.tensor = None) -> torch.tensor:
        # extract first token tensor
        return emb[:, 0]
        # return emb

    @overrides
    def get_input_dim(self) -> int:
        raise NotImplementedError

    @overrides
    def get_output_dim(self) -> int:
        return self.bert_dim
