from config.common import *


class StanceCfg(DirCfg):
    corpus_name = 'Stance'
    max_vocab_size = 100000
    max_seq_len = 30
    hp = HP()
    # hp.phase = DirCfg.phase_real_str
    # hp.phase = DirCfg.phase_gan_str
    hp.phase = DirCfg.phase_fake_str
    hp.file_ratio = 0.5
    hp.batch_size = 16
    hp.max_pieces = 128
    hp.patience = 3

    hp.training_size = {
        0.05: 129,
        0.1: 260,
        0.2: 523,
        0.5: 1309,
        1: 2621,
    }
    hp.batch_per_epoch = int(hp.training_size[hp.file_ratio] / hp.batch_size) + 1
    if hp.phase == DirCfg.phase_real_str:               # real
        hp.lr = 1e-5
    elif hp.phase == DirCfg.phase_gan_str:              # gan
        hp.batch_per_epoch = 5 * hp.batch_per_epoch
        hp.lr = 1e-5
    elif hp.phase == DirCfg.phase_fake_str:             # fake
        hp.lr = 1e-7
        hp.gen_step = 2 + 1
    else:
        raise ValueError('Phase name not found.')

    if 'C:' in DirCfg.home:
        root = os.path.join(DirCfg.home, 'OneDrive/data61/project/%s/dataset/%s'
                            % (DirCfg.project_name, corpus_name))
        root_local = os.path.join(DirCfg.home, 'Documents/data61/project/%s/dataset/%s'
                                  % (DirCfg.project_name, corpus_name))
    elif 'home' in DirCfg.home:
        root = '/home/xu052/text_classification/dataset/stance/'
        root_local = ''
    else:
        root = ''
        root_local = ''

    target = ['a', 'cc', 'fm', 'hc', 'la']

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

    dev_path = os.path.join(data_dir, 'dev.txt')

    test_raw_path = os.path.join(data_dir, 'SemEval2016-Task6-subtaskA-test-%s.txt')
    test_raw_path_all_target = os.path.join(data_dir, 'SemEval2016-Task6-subtaskA-test-all.txt')
    test_path = os.path.join(data_dir, 'test.txt')

    train_real_meta_path = os.path.join(result_dir, 'train_real_meta_%s_%sp.th')
    train_gan_meta_path = os.path.join(result_dir, 'train_gan_meta_%s_%sp.th')
    train_fake_meta_path = os.path.join(result_dir, 'train_fake_meta_%s_%sp.th')
    test_meta_path = os.path.join(result_dir, 'test_meta_%s_%sp.th')

    img_real_feature_path = 'real_features.png'
    img_gen_feature_path = 'gen_features.png'
    img_fake_feature_path = 'fake_features.png'
