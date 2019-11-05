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

from my_library.models import BaseLSTM
from my_library.dataset_readers.norm_yahoo import YahooDatasetReader
from config import YahooConfig

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

torch.manual_seed(2019)

# n_label = YahooConfig.n_label
lr = 0.0001
epochs = 50
d_word_emb = 300
d_rnn = 128
d_hidden = 128
patience = 10
batch_size = 64
dropout = 0.1
config_file = YahooConfig

reader = YahooDatasetReader(tokenizer=lambda string: string.split()[:config_file.max_seq_len],
                            token_indexers={"tokens": SingleIdTokenIndexer()}
                            )

train_dataset = reader.read(cached_path(config_file.train_ratio_path % '1'))
validation_dataset = reader.read(cached_path(config_file.dev_ratio_path % '1'))
# test_dataset = reader.read(cached_path(config_file.dev_ratio_path % '1'))

vocab = Vocabulary.from_instances(train_dataset,
                                  # min_count={'tokens': 2},
                                  only_include_pretrained_words=True,
                                  max_vocab_size=config_file.max_vocab_size,
                                  pretrained_files={'tokens': config_file.GLOVE_840B_300D},
                                  )

# vocab.save_to_files("/tmp/vocabulary")

logger.info('Vocab size: %s' % vocab.get_vocab_size())

iterator = BucketIterator(batch_size=batch_size,
                          biggest_batch_first=True,
                          sorting_keys=[('tokens', 'num_tokens')],
                          )
iterator.index_with(vocab)

word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder(
    {"tokens": Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                         embedding_dim=d_word_emb)})

encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(
    input_size=d_word_emb,
    hidden_size=d_rnn,
    num_layers=2,
    dropout=dropout,
    bidirectional=True, batch_first=True))

model = BaseLSTM(
    vocab,
    word_embeddings,
    encoder,
    d_hidden,
    dropout
)

if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1

trainer = Trainer(
    model=model,
    optimizer=optim.Adam(model.parameters(), lr=lr),
    iterator=iterator,
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    cuda_device=cuda_device,
    num_epochs=epochs,
    patience=patience,
    serialization_dir=config_file.model_path
)

metrics = trainer.train()
