import os


class DirConfig(object):
    home = str(os.path.expanduser('~'))
    if 'C:' in home:
        W2V_DIR = os.path.join(home, 'Downloads/dataset/word_vec/')
        GLOVE_840B_300D = os.path.join(W2V_DIR, 'glove.840B.300d.txt')
        BERT_VOC = os.path.join(home, 'Downloads/dataset/bert/bert-base-uncased-vocab.txt')
        BERT_MODEL = os.path.join(home, 'Downloads/dataset/bert/bert-base-uncased.tar.gz')
    elif 'home' in home:
        W2V_DIR = '/home/xu052/glove/'
        GLOVE_840B_300D = ''
        BERT_VOC = ''
        BERT_MODEL = ''
    else:
        W2V_DIR = '/Users/xu052/Documents/project/glove/'
        GLOVE_840B_300D = ''
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

    vocab_path = os.path.join(cache_dir, 'vocab')
    model_path = os.path.join(model_dir, 'model.th')
    best_model_path = os.path.join(model_path, 'best.th')

    train_raw_path = os.path.join(data_dir, 'train_raw.csv')
    train_ratio_path = os.path.join(data_dir, 'train_%sp.csv')
    dev_ratio_path = os.path.join(data_dir, 'dev_%sp.csv')
    test_raw_path = os.path.join(data_dir, 'test_raw.csv')
    test_path = os.path.join(data_dir, 'test.csv')


class StanceConfig(DirConfig):
    corpus_name = 'Stance'
    max_vocab_size = 100000
    max_seq_len = 30

    hparam = {
        'a': {
            'lr': 0.0001,
            'epochs': 999,
            'patience': 5,
            'batch_size': 8,
            'd_hidden': 100,
            'dropout': 0.3,
            'lambda': 0,
            'cuda_device': 0,
            'file_frac': 100,
        }
    }

    if 'C:' in DirConfig.home:
        root = os.path.join(DirConfig.home, 'OneDrive/data61/project/text_classification/dataset/stance')
    elif 'home' in DirConfig.home:
        root = '/home/xu052/text_classification/dataset/stance/'
    else:
        root = ''

    target = ['a', 'cc', 'fm', 'hc', 'la']

    labels = [
        'FAVOR',
        'AGAINST',
        'NONE'
    ]

    n_label = len(labels)

    data_dir = os.path.join(root, 'data')
    cache_dir = os.path.join(root, 'cache')
    model_dir = os.path.join(root, 'models')
    result_dir = os.path.join(root, 'results')

    vocab_path = os.path.join(cache_dir, 'vocab')
    model_path = os.path.join(model_dir, 'model_%s.th')

    train_raw_path = os.path.join(data_dir, 'semeval2016-task6-subtaskA-train-dev-%s.txt')
    train_ratio_path = os.path.join(data_dir, 'train_%s_%sp.txt')
    dev_ratio_path = os.path.join(data_dir, 'dev_%s_%sp.txt')
    test_raw_path = os.path.join(data_dir, 'SemEval2016-Task6-subtaskA-test-%s.txt')
    test_path = os.path.join(data_dir, 'test_%s.txt')
