import os


class DirConfig(object):
    project_name = 'text_classification'
    home = str(os.path.expanduser('~'))
    if 'C:' in home:
        W2V_DIR = os.path.join(home, 'Downloads/dataset/word_vec/')
        GLOVE_840B_300D = os.path.join(W2V_DIR, 'glove.840B.300d.txt')
        GLOVE_TWITTER_27B_200D = os.path.join(W2V_DIR, 'glove.twitter.27B.200d.txt')
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

    phase_real_str = 'cls_on_real'
    phase_gan_str = 'gan'
    phase_fake_str = 'cls_on_fake'


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
    max_vocab_size = 10000
    max_seq_len = 30

    class HP:
        # phase = 'cls_on_real'
        phase = 'gan'
        # phase = 'cls_on_fake'

        training_size = {
            '5': 129,
            '10': 260,
            '20': 523,
            '40': 1047,
            '80': 2096,
            '100': 2621,
        }
        file_ratio = 40
        lr = 0.00001
        patience = 15
        gen_step = 64 + 1

        n_epoch_gan = 500
        conservative_rate = 0.1
        batch_size = 16
        alpha = 1
        if phase == DirConfig.phase_gan_str:
            alpha = 5
        batch_per_epoch = alpha * (int(training_size[str(file_ratio)] / batch_size) + 1)
        batch_per_generator = 20
        d_hidden = 768
        dropout = 0.1
        cuda_device = 0

    if 'C:' in DirConfig.home:
        root = os.path.join(DirConfig.home, 'OneDrive/data61/project/%s/dataset/%s'
                            % (DirConfig.project_name, corpus_name))
        root_local = os.path.join(DirConfig.home, 'Documents/data61/project/%s/dataset/%s'
                                  % (DirConfig.project_name, corpus_name))
    elif 'home' in DirConfig.home:
        root = '/home/xu052/text_classification/dataset/stance/'
        root_local = ''
    else:
        root = ''
        root_local = ''

    target = ['a', 'cc', 'fm', 'hc', 'la']

    labels = [
        'FAVOR',
        'AGAINST',
        'NONE'
    ]

    n_label = len(labels)

    data_dir = os.path.join(root, 'data')
    cache_dir = os.path.join(root, 'cache')
    model_dir = os.path.join(root_local, 'models')
    result_dir = os.path.join(root_local, 'results')

    vocab_path = os.path.join(cache_dir, 'vocab')
    model_path = os.path.join(model_dir, 'model_%s.th')

    train_raw_path = os.path.join(data_dir, 'semeval2016-task6-subtaskA-train-dev-%s.txt')
    train_raw_path_all_target = os.path.join(data_dir, 'semeval2016-task6-subtaskA-train-dev-all.txt')
    train_path = os.path.join(data_dir, 'train_100p.txt')
    train_ratio_path = os.path.join(data_dir, 'train_%sp.txt')

    dev_path = os.path.join(data_dir, 'dev.txt')
    dev_ratio_path = os.path.join(data_dir, 'dev_%sp.txt')

    test_raw_path = os.path.join(data_dir, 'SemEval2016-Task6-subtaskA-test-%s.txt')
    test_raw_path_all_target = os.path.join(data_dir, 'SemEval2016-Task6-subtaskA-test-all.txt')
    test_path = os.path.join(data_dir, 'test.txt')

    train_real_meta_path = os.path.join(result_dir, 'train_real_meta_%s_%sp.th')
    train_gan_meta_path = os.path.join(result_dir, 'train_gan_meta_%s_%sp.th')
    train_fake_meta_path = os.path.join(result_dir, 'train_fake_meta_%s_%sp.th')
    test_meta_path = os.path.join(result_dir, 'test_meta_%s_%sp.th')

    img_gen_feature_path = 'gen_feature_%s.png'


class TRECConfig(DirConfig):
    corpus_name = 'TREC'
    max_vocab_size = 10000
    max_seq_len = 30

    hparam = {
        'lr': 0.00001,
        'patience': 15,
        'conservative_rate': 0.5,
        'batch_size': 16,
        'batch_per_epoch': 20,
        'batch_per_generator': 30,
        'gen_step': 10,
        'd_hidden': 768,
        'dropout': 0.1,
        'cuda_device': 0,
        'file_frac': 5,
    }

    if 'C:' in DirConfig.home:
        root = os.path.join(DirConfig.home, 'OneDrive/data61/project/%s/dataset/%s'
                            % (DirConfig.project_name, corpus_name))
        root_local = os.path.join(DirConfig.home, 'Documents/data61/project/%s/dataset/%s'
                                  % (DirConfig.project_name, corpus_name))
    elif 'home' in DirConfig.home:
        root = '/home/xu052/text_classification/dataset/TREC/'
        root_local = ''
    else:
        root = ''
        root_local = ''

    # 6 classes
    labels = [
        'NUM', 'DESC', 'HUM', 'LOC', 'ENTY', 'ABBR'
    ]

    n_label = len(labels)

    data_dir = os.path.join(root, 'data')
    cache_dir = os.path.join(root, 'cache')
    model_dir = os.path.join(root_local, 'models')
    result_dir = os.path.join(root_local, 'results')

    vocab_path = os.path.join(cache_dir, 'vocab')

    train_raw_path = os.path.join(data_dir, 'train.txt')
    train_norm_path = os.path.join(data_dir, 'train_norm.txt')
    train_ratio_path = os.path.join(data_dir, 'train_%sp.txt')

    dev_ratio_path = os.path.join(data_dir, 'dev_%sp.txt')

    test_raw_path = os.path.join(data_dir, 'test.txt')
    test_norm_path = os.path.join(data_dir, 'test_norm.txt')
    test_path = os.path.join(data_dir, 'test_%sp.txt')

    train_real_meta_path = os.path.join(result_dir, 'train_real_meta_%s_%sp.th')
    train_gan_meta_path = os.path.join(result_dir, 'train_gan_meta_%s_%sp.th')
    train_fake_meta_path = os.path.join(result_dir, 'train_fake_meta_%s_%sp.th')
    test_meta_path = os.path.join(result_dir, 'test_meta_%s_%sp.th')
    img_gen_feature_path = os.path.join(result_dir, 'img', 'gen_feature_%s.png')


