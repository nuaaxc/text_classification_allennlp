import os


class DirConfig(object):
    home = str(os.path.expanduser('~'))
    if 'C:' in home:
        W2V_DIR = os.path.join(home, 'Downloads/dataset/word_vec/')
        BERT_VOC = os.path.join(home, 'Downloads/dataset/bert/bert-large-uncased-vocab.txt')
        BERT_MODEL = os.path.join(home, 'Downloads/dataset/bert/bert-large-uncased.tar.gz')
    elif 'home' in home:
        W2V_DIR = '/home/xu052/glove/'
        BERT_VOC = ''
        BERT_MODEL = ''
    else:
        W2V_DIR = '/Users/xu052/Documents/project/glove/'
        BERT_VOC = ''
        BERT_MODEL = ''


class DBPediaConfig(DirConfig):
    corpus_name = 'DBPedia'

    if 'C:' in DirConfig.home:
        root = os.path.join(DirConfig.home, 'OneDrive/data61/project/text_classification/dataset/dbpedia')
    elif 'home' in DirConfig.home:
        root = '/home/xu052/text_classification/dataset/dbpedia/'

    labels = ['Company',
              'EducationalInstitution',
              'Artist',
              'Athlete',
              'OfficeHolder',
              'MeanOfTransportation',
              'Building',
              'NaturalPlace',
              'Village',
              'Animal',
              'Plant',
              'Album',
              'Film',
              'WrittenWork']


class YahooConfig(DirConfig):
    corpus_name = 'Yahoo'
    max_vocab_size = 100000
    max_seq_len = 100

    if 'C:' in DirConfig.home:
        root = os.path.join(DirConfig.home, 'OneDrive/data61/project/text_classification/dataset/yahoo')
    elif 'home' in DirConfig.home:
        root = '/home/xu052/text_classification/dataset/yahoo/'
    else:
        root = ''

    labels = ['Society & Culture',
              'Science & Mathematics',
              'Health',
              'Education & Reference',
              'Computers & Internet',
              'Sports',
              'Business & Finance',
              'Entertainment & Music',
              'Family & Relationships',
              'Politics & Government']
    n_label = len(labels)

    data_dir = os.path.join(root, 'data')
    cache_dir = os.path.join(root, 'cache')
    model_dir = os.path.join(root, 'models')
    result_dir = os.path.join(root, 'results')

    train_path = os.path.join(data_dir, 'train.csv')
    train_ratio_path = os.path.join(data_dir, 'train_%sp.csv')
    dev_ratio_path = os.path.join(data_dir, 'dev_%sp.csv')
    test_path = os.path.join(data_dir, 'test.csv')

