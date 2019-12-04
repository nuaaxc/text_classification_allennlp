import os


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


class YelpFullConfig(DirConfig):
    corpus_name = 'YelpFull'
    max_vocab_size = 10000
    max_seq_len = 30

    hp = HP()
    hp.phase = DirConfig.phase_real_str
    # hp.phase = 'gan'
    # hp.phase = 'cls_on_fake'
    hp.file_ratio = 0.01
    hp.batch_size = 16
    hp.training_size = {
        0.001: 580,
        0.005: 2920,
        0.01: 5845,
        0.05: 29245,
        0.1: 58495,
        0.2: 116995,
        0.5: 292491,
        1: 584986,  # 585000
    }
    hp.batch_per_epoch = int(hp.training_size[hp.file_ratio] / hp.batch_size) + 1

    if 'C:' in DirConfig.home:
        root = os.path.join(DirConfig.home, 'OneDrive/data61/project/%s/dataset/%s'
                            % (DirConfig.project_name, corpus_name))
        root_local = os.path.join(DirConfig.home, 'Documents/data61/project/%s/dataset/%s'
                                  % (DirConfig.project_name, corpus_name))
    elif 'home' in DirConfig.home:
        root = '/home/xu052/%s/dataset/%s/' % (DirConfig.project_name, corpus_name)
        root_local = ''
    else:
        root = ''
        root_local = ''

    labels = ['1', '2', '3', '4', '5']

    n_label = len(labels)

    data_dir = os.path.join(root, 'data')
    cache_dir = os.path.join(root, 'cache')
    model_dir = os.path.join(root_local, 'models')
    result_dir = os.path.join(root_local, 'results')

    vocab_path = os.path.join(cache_dir, 'vocab')
    model_path = os.path.join(model_dir, 'model_%s.th')

    train_raw_path = os.path.join(data_dir, 'train_raw.csv')
    train_norm_path = os.path.join(data_dir, 'train_norm.txt')
    train_path = os.path.join(data_dir, 'train_1p.txt')
    train_ratio_path = os.path.join(data_dir, 'train_%sp.txt')

    dev_path = os.path.join(data_dir, 'dev.txt')

    test_raw_path = os.path.join(data_dir, 'test_raw.csv')
    test_path = os.path.join(data_dir, 'test.txt')

    train_real_meta_path = os.path.join(result_dir, 'train_real_meta_%s_%sp.th')
    train_gan_meta_path = os.path.join(result_dir, 'train_gan_meta_%s_%sp.th')
    train_fake_meta_path = os.path.join(result_dir, 'train_fake_meta_%s_%sp.th')
    test_meta_path = os.path.join(result_dir, 'test_meta_%s_%sp.th')

    img_gen_feature_path = 'gen_feature_%s.png'


class NGConfig(DirConfig):
    corpus_name = 'Newsgroups'
    max_vocab_size = 10000
    max_seq_len = 30

    hp = HP()
    hp.phase = DirConfig.phase_real_str
    # hp.phase = 'gan'
    # hp.phase = 'cls_on_fake'
    hp.file_ratio = 0.1
    hp.batch_size = 64
    hp.max_pieces = 256
    hp.lr = 0.001
    hp.training_size = {
        0.05: 496,
        0.1: 1007,
        0.2: 2023,
        0.5: 5073,
        1: 10155,
    }
    hp.batch_per_epoch = int(hp.training_size[hp.file_ratio] / hp.batch_size) + 1

    if 'C:' in DirConfig.home:
        root = os.path.join(DirConfig.home, 'OneDrive/data61/project/%s/dataset/%s'
                            % (DirConfig.project_name, corpus_name))
        root_local = os.path.join(DirConfig.home, 'Documents/data61/project/%s/dataset/%s'
                                  % (DirConfig.project_name, corpus_name))
    elif 'home' in DirConfig.home:
        root = '/home/xu052/%s/dataset/%s/' % (DirConfig.project_name, corpus_name)
        root_local = ''
    else:
        root = ''
        root_local = ''

    labels = ['alt.atheism',
              'comp.graphics',
              'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware',
              'comp.windows.x',
              'misc.forsale',
              'rec.autos',
              'rec.motorcycles',
              'rec.sport.baseball',
              'rec.sport.hockey',
              'sci.crypt',
              'sci.electronics',
              'sci.med',
              'sci.space',
              'soc.religion.christian',
              'talk.politics.guns',
              'talk.politics.mideast',
              'talk.politics.misc',
              'talk.religion.misc']

    n_label = len(labels)

    data_dir = os.path.join(root, 'data')
    cache_dir = os.path.join(root, 'cache')
    model_dir = os.path.join(root_local, 'models')
    result_dir = os.path.join(root_local, 'results')

    train_norm_path = os.path.join(data_dir, 'train_norm.txt')
    train_path = os.path.join(data_dir, 'train_1p.txt')
    train_ratio_path = os.path.join(data_dir, 'train_%sp.txt')
    ground_truth_path = os.path.join(data_dir, 'labels.csv')

    dev_path = os.path.join(data_dir, 'dev.txt')

    test_path = os.path.join(data_dir, 'test.txt')

    train_real_meta_path = os.path.join(result_dir, 'train_real_meta_%s_%sp.th')
    train_gan_meta_path = os.path.join(result_dir, 'train_gan_meta_%s_%sp.th')
    train_fake_meta_path = os.path.join(result_dir, 'train_fake_meta_%s_%sp.th')
    test_meta_path = os.path.join(result_dir, 'test_meta_%s_%sp.th')

    img_gen_feature_path = 'gen_feature_%s.png'


class AffectConfig(DirConfig):
    corpus_name = 'Affect'
    max_vocab_size = 10000
    max_seq_len = 30

    hp = HP()
    hp.lr = 0.0001
    hp.phase = DirConfig.phase_real_str
    # hp.phase = 'gan'
    # hp.phase = 'cls_on_fake'
    hp.file_ratio = 1
    hp.batch_size = 8
    hp.training_size = {
        0.1: 114,
        0.2: 233,
        0.4: 469,
        0.8: 942,
        1: 1181,
    }
    hp.batch_per_epoch = int(hp.training_size[hp.file_ratio] / hp.batch_size) + 1

    if 'C:' in DirConfig.home:
        root = os.path.join(DirConfig.home, 'OneDrive/data61/project/%s/dataset/%s'
                            % (DirConfig.project_name, corpus_name))
        root_local = os.path.join(DirConfig.home, 'Documents/data61/project/%s/dataset/%s'
                                  % (DirConfig.project_name, corpus_name))
    elif 'home' in DirConfig.home:
        root = '/home/xu052/%s/dataset/%s/' % (DirConfig.project_name, corpus_name)
        root_local = ''
    else:
        root = ''
        root_local = ''

    labels = [-3, -2, -1, 0, 1, 2, 3]

    n_label = len(labels)

    data_dir = os.path.join(root, 'data')
    cache_dir = os.path.join(root, 'cache')
    model_dir = os.path.join(root_local, 'models')
    result_dir = os.path.join(root_local, 'results')

    model_path = os.path.join(model_dir, 'model_%s.th')

    train_raw_path = os.path.join(data_dir, 'train_raw.txt')
    train_path = os.path.join(data_dir, 'train_1p.txt')
    train_ratio_path = os.path.join(data_dir, 'train_%sp.txt')

    dev_raw_path = os.path.join(data_dir, 'dev_raw.txt')
    dev_path = os.path.join(data_dir, 'dev.txt')

    test_raw_path = os.path.join(data_dir, 'test_raw.txt')
    test_path = os.path.join(data_dir, 'test.txt')

    train_real_meta_path = os.path.join(result_dir, 'train_real_meta_%s_%sp.th')
    train_gan_meta_path = os.path.join(result_dir, 'train_gan_meta_%s_%sp.th')
    train_fake_meta_path = os.path.join(result_dir, 'train_fake_meta_%s_%sp.th')
    test_meta_path = os.path.join(result_dir, 'test_meta_%s_%sp.th')

    img_gen_feature_path = 'gen_feature_%s.png'


class AGConfig(DirConfig):
    corpus_name = 'AG'
    max_vocab_size = 10000
    max_seq_len = 30

    hp = HP()
    hp.phase = DirConfig.phase_real_str
    # hp.phase = 'gan'
    # hp.phase = 'cls_on_fake'
    hp.file_ratio = 0.1
    hp.batch_size = 16
    hp.training_size = {
        0.001: 108,
        0.005: 540,
        0.01: 1080,
        0.05: 5400,
        0.1: 10800,
        0.2: 21600,
        0.5: 54000,
        1: 108000,
    }
    hp.batch_per_epoch = int(hp.training_size[hp.file_ratio] / hp.batch_size) + 1

    if 'C:' in DirConfig.home:
        root = os.path.join(DirConfig.home, 'OneDrive/data61/project/%s/dataset/%s'
                            % (DirConfig.project_name, corpus_name))
        root_local = os.path.join(DirConfig.home, 'Documents/data61/project/%s/dataset/%s'
                                  % (DirConfig.project_name, corpus_name))
    elif 'home' in DirConfig.home:
        root = '/home/xu052/%s/dataset/%s/' % (DirConfig.project_name, corpus_name)
        root_local = ''
    else:
        root = ''
        root_local = ''

    labels = [
        'World',
        'Sports',
        'Business',
        'SciTech'
    ]

    n_label = len(labels)

    data_dir = os.path.join(root, 'data')
    cache_dir = os.path.join(root, 'cache')
    model_dir = os.path.join(root_local, 'models')
    result_dir = os.path.join(root_local, 'results')

    model_path = os.path.join(model_dir, 'model_%s.th')

    train_raw_path = os.path.join(data_dir, 'train_raw.csv')
    train_norm_path = os.path.join(data_dir, 'train_norm.txt')
    train_path = os.path.join(data_dir, 'train_1p.txt')
    train_ratio_path = os.path.join(data_dir, 'train_%sp.txt')

    dev_path = os.path.join(data_dir, 'dev.txt')

    test_raw_path = os.path.join(data_dir, 'test_raw.csv')
    test_path = os.path.join(data_dir, 'test.txt')

    train_real_meta_path = os.path.join(result_dir, 'train_real_meta_%s_%sp.th')
    train_gan_meta_path = os.path.join(result_dir, 'train_gan_meta_%s_%sp.th')
    train_fake_meta_path = os.path.join(result_dir, 'train_fake_meta_%s_%sp.th')
    test_meta_path = os.path.join(result_dir, 'test_meta_%s_%sp.th')

    img_gen_feature_path = 'gen_feature_%s.png'

