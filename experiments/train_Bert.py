import logging
import torch
import torch.optim as optim

from allennlp.common.file_utils import cached_path

from allennlp.data.token_indexers import PretrainedBertIndexer, SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper

from my_library.models import BertSentencePooling
from my_library.models import BaseLSTM
from my_library.dataset_readers.yahoo import YahooDatasetReader
from config import YahooConfig

torch.manual_seed(2019)

n_label = YahooConfig.n_label
lr = 0.001
epochs = 50
patience = 10
batch_size = 64
config_file = YahooConfig

token_indexer = PretrainedBertIndexer(
    pretrained_model=config_file.BERT_VOC,
    # max_pieces=config_file.max_seq_len,
    do_lowercase=True,
)

reader = YahooDatasetReader(
    tokenizer=lambda string: token_indexer.wordpiece_tokenizer(string)[:config_file.max_seq_len],
    token_indexers={"tokens": token_indexer}
)

train_dataset = reader.read(cached_path(config_file.train_ratio_path % '1'))
validation_dataset = reader.read(cached_path(config_file.dev_ratio_path % '1'))
# test_dataset = reader.read(cached_path(config_file.dev_ratio_path % '1'))

vocab = Vocabulary.from_instances(train_dataset,
                                  # min_count={'tokens': 2},
                                  # only_include_pretrained_words=True,
                                  max_vocab_size=config_file.max_vocab_size
                                  )

iterator = BucketIterator(batch_size=batch_size,
                          biggest_batch_first=True,
                          sorting_keys=[('tokens', 'num_tokens')],
                          )
iterator.index_with(vocab)

bert_embedder = PretrainedBertEmbedder(
    pretrained_model=config_file.BERT_MODEL,
    # requires_grad=True,
    top_layer_only=True,  # conserve memory
)
word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": bert_embedder},
                                                            allow_unmatched_keys=True)

encoder = BertSentencePooling(vocab, word_embeddings.get_output_dim())

model = BaseLSTM(
    vocab,
    n_label,
    word_embeddings,
    encoder,
)

model = model.cuda(0)

trainer = Trainer(
    model=model,
    optimizer=optim.Adam(model.parameters(), lr=lr),
    iterator=iterator,
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    cuda_device=0,
    num_epochs=epochs,
    patience=patience,
)

metrics = trainer.train()
