from config.common import *


class StanceCfg(DirCfg):
    corpus_name = 'Stance'
    max_vocab_size = 100000
    max_seq_len = 30

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

    target = ['a', 'cc', 'fm', 'hc', 'la']

    training_size = {
        0.01: 24,
        0.02: 51,
        0.05: 129,
        0.1: 260,
        0.2: 523,
        0.5: 1309,
        1: 2621,
    }

    labels = [
        '__label__FAVOR',
        '__label__AGAINST',
        '__label__NONE'
    ]

    n_label = len(labels)

    data_dir = os.path.join(root, 'data')
    cache_dir = os.path.join(root, 'cache')
    model_dir = os.path.join(root_local, 'models')
    result_dir = os.path.join(root_local, 'results')

    model_path = os.path.join(model_dir, 'model_%s.th')

    train_raw_path = os.path.join(data_dir, 'semeval2016-task6-subtaskA-train-dev-%s.txt')
    train_raw_path_all_target = os.path.join(data_dir, 'semeval2016-task6-subtaskA-train-dev-all.txt')
    train_norm_path = os.path.join(data_dir, 'train_norm.txt')
    train_path = os.path.join(data_dir, 'train_1p.txt')
    train_ratio_path = os.path.join(data_dir, 'train_%sp.txt')
    eda_train_ratio_path = os.path.join(data_dir, 'eda_train_%sp.txt')

    dev_path = os.path.join(data_dir, 'dev.txt')

    test_raw_path = os.path.join(data_dir, 'SemEval2016-Task6-subtaskA-test-%s.txt')
    test_raw_path_all_target = os.path.join(data_dir, 'SemEval2016-Task6-subtaskA-test-all.txt')
    test_path = os.path.join(data_dir, 'test.txt')

    train_real_meta_path = os.path.join(result_dir, 'train_real_meta_%s_%sp.th')
    train_gan_meta_path = os.path.join(result_dir, 'train_gan_meta_%s_%sp.th')
    train_fake_meta_path = os.path.join(result_dir, 'train_fake_meta_%s_%sp.th')
    test_meta_path = os.path.join(result_dir, 'test_meta_%s_%sp.th')

    img_real_feature_path = 'real_features_%s_%s.png'
    img_gen_feature_path = 'gen_features_%s_%s.png'
    img_fake_feature_path = 'fake_features_%s_%s.png'

    img_quant_path = os.path.join(result_dir, 'quant_%s.png' % corpus_name)


class StanceCfgBert(StanceCfg):
    hp = HP()
    # hp.phase = DirCfg.phase_real_str
    # hp.phase = DirCfg.phase_gan_str
    hp.phase = DirCfg.phase_fake_str
    hp.file_ratio = 1
    hp.max_pieces = 128
    hp.patience = 5
    hp.cuda_device = 0

    if hp.phase == DirCfg.phase_real_str:  # real
        hp.lr = 1e-5
        hp.batch_size = 16
        hp.tune_bert = True
    elif hp.phase == DirCfg.phase_gan_str:  # gan
        hp.lr = 1e-5
        hp.n_epoch_gan = 800
        # hp.batch_size = 128
        hp.batch_size = 16
        hp.conservative_rate = 0.0
    elif hp.phase == DirCfg.phase_fake_str:  # fake
        hp.patience = 3
        hp.lr = 1e-5
        # hp.lr = 1e-8
        hp.gen_step = 1 + 16
        hp.batch_size = 64
    else:
        raise ValueError('Phase name not found.')

    hp.batch_per_epoch = int(StanceCfg.training_size[hp.file_ratio] / hp.batch_size) + 1


class StanceCfgLSTM(StanceCfg):
    model_name = 'lstm'
    hp = HP()
    hp.phase = DirCfg.phase_real_str
    hp.file_ratio = 1
    hp.patience = 3
    hp.cuda_device = 0
    hp.lr = 1e-4
    hp.batch_size = 16
    hp.d_rnn = 64
    hp.d_dense = 128


class StanceCfgCNN(StanceCfg):
    model_name = 'cnn'
    hp = HP()
    hp.phase = DirCfg.phase_real_str
    hp.file_ratio = 0.1
    hp.patience = 3
    hp.cuda_device = 0
    hp.lr = 1e-4
    hp.batch_size = 16
    hp.filter_width = 5
    hp.num_filters = 128
    hp.d_dense = 128
