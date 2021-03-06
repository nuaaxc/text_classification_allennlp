from config.common import *


class OffensiveCfg(DirCfg):
    corpus_name = 'Offensive'
    max_vocab_size = 10000
    max_seq_len = 30
    hp = HP()
    # hp.phase = DirCfg.phase_real_str
    # hp.phase = DirCfg.phase_gan_str
    hp.phase = DirCfg.phase_fake_str
    hp.file_ratio = 1
    hp.max_pieces = 128
    hp.patience = 5
    hp.cuda_device = 0

    hp.training_size = {
        0.01: 118,
        0.02: 238,
        0.05: 595,
        0.1: 1191,
        0.2: 2383,
        0.5: 5958,
        1: 11911,
    }

    if hp.phase == DirCfg.phase_real_str:               # real
        hp.lr = 1e-5
        hp.tune_bert = True
        hp.batch_size = 16
    elif hp.phase == DirCfg.phase_gan_str:              # gan
        hp.lr = 1e-5
        hp.n_epoch_gan = 800
        hp.batch_size = 128
        # hp.batch_size = 16
        hp.conservative_rate = 0.0
    elif hp.phase == DirCfg.phase_fake_str:             # fake
        hp.patience = 3
        # hp.lr = 1e-7
        hp.lr = 1e-5
        hp.gen_step = 1 + 4
        hp.batch_size = 256
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

    labels = ['OFF', 'NOT']

    n_label = len(labels)

    data_dir = os.path.join(root, 'data')
    cache_dir = os.path.join(root, 'cache')
    model_dir = os.path.join(root_local, 'models')
    result_dir = os.path.join(root_local, 'results')

    train_raw_path = os.path.join(data_dir, 'train_raw.tsv')
    train_norm_path = os.path.join(data_dir, 'train_norm.txt')
    train_path = os.path.join(data_dir, 'train_1p.txt')
    train_ratio_path = os.path.join(data_dir, 'train_%sp.txt')
    ground_truth_path = os.path.join(data_dir, 'labels.csv')

    dev_path = os.path.join(data_dir, 'dev.txt')

    test_raw_path = os.path.join(data_dir, 'test_raw.tsv')
    test_path = os.path.join(data_dir, 'test.txt')

    train_real_meta_path = os.path.join(result_dir, 'train_real_meta_%s_%sp.th')
    train_gan_meta_path = os.path.join(result_dir, 'train_gan_meta_%s_%sp.th')
    train_fake_meta_path = os.path.join(result_dir, 'train_fake_meta_%s_%sp.th')
    test_meta_path = os.path.join(result_dir, 'test_meta_%s_%sp.th')

    img_real_feature_path = 'real_features_%s_%s.png'
    img_gen_feature_path = 'gen_features_%s_%s.png'
    img_fake_feature_path = 'fake_features_%s_%s.png'

    img_quant_path = os.path.join(result_dir, 'quant_%s.png' % corpus_name)
