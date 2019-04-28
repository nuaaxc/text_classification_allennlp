import torch

from allennlp.common.file_utils import cached_path
from my_library.dataset_readers.dbpedia import DBPediaDatasetReader

from config import DBPediaConfig

torch.manual_seed(2019)

reader = DBPediaDatasetReader()

# train_dataset = reader.read(cached_path(DBPediaConfig.train_ratio_path % '10'))
validation_dataset = reader.read(cached_path(DBPediaConfig.dev_ratio_path % '10'))


for instance in validation_dataset:
    print(instance)