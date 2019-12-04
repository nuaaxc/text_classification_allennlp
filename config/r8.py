from config.common import *


class R8Cfg(DirCfg):
    corpus_name = 'R8'
    max_vocab_size = 10000
    max_seq_len = 30
    hp = HP()
    # hp.phase = DirCfg.phase_real_str
    # hp.phase = DirCfg.phase_gan_str
    hp.phase = DirCfg.phase_fake_str
    hp.file_ratio = 0.05
    hp.max_pieces = 128
    hp.patience = 5
    hp.cuda_device = 0

    hp.training_size = {
        0.05: 242,
        0.1: 489,
        0.2: 985,
        0.5: 2464,
        1: 4933,
    }

    if hp.phase == DirCfg.phase_real_str:               # real
        hp.lr = 1e-5
        hp.batch_size = 16
        hp.tune_bert = True
    elif hp.phase == DirCfg.phase_gan_str:              # gan
        hp.lr = 5e-4
        hp.n_epoch_gan = 800
        # hp.batch_size = 128
        hp.batch_size = 16
        hp.conservative_rate = 0.8
    elif hp.phase == DirCfg.phase_fake_str:             # fake
        hp.patience = 5
        hp.lr = 1e-5
        # hp.lr = 1e-7
        hp.gen_step = 1 + 64
        hp.batch_size = 16
    else:
        raise ValueError('Phase name not found.')

    hp.batch_per_epoch = int(hp.training_size[hp.file_ratio] / hp.batch_size) + 1

    if 'C:' in DirCfg.home:
        root = os.path.join(DirCfg.home, 'OneDrive/data61/project/%s/dataset/%s'
                            % (DirCfg.project_name, corpus_name))
        root_local = os.path.join(DirCfg.home, 'Documents/data61/project/%s/dataset/%s'
                                  % (DirCfg.project_name, corpus_name))
    elif 'home' in DirCfg.home:
        root = '/home/xu052/%s/dataset/%s/' % (DirCfg.project_name, corpus_name)
        root_local = ''
    else:
        root = ''
        root_local = ''

    # labels = ['__label__acq',
    #           '__label__ship',
    #           '__label__grain',
    #           '__label__interest',
    #           '__label__crude',
    #           '__label__earn',
    #           '__label__money-fx',
    #           '__label__trade']

    labels = ['acq',
              'ship',
              'grain',
              'interest',
              'crude',
              'earn',
              'money-fx',
              'trade']

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

    img_real_feature_path = 'real_features.png'
    img_gen_feature_path = 'gen_features.png'
    img_fake_feature_path = 'fake_features.png'

    img_quant_path = os.path.join(result_dir, 'quant_%s.png' % corpus_name)

