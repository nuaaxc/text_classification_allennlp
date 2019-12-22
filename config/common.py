import os


class DirCfg(object):
    project_name = 'text_classification'
    home = str(os.path.expanduser('~'))
    if 'C:' in home:
        W2V_DIR = os.path.join(home, 'Downloads/dataset/word_vec/')
        GLOVE_840B_300D = os.path.join(W2V_DIR, 'glove.840B.300d.txt')
        GLOVE_840B_300D_DIM = 300
        GLOVE_TWITTER_27B_200D = os.path.join(W2V_DIR, 'glove.twitter.27B.200d.txt')
        GLOVE_TWITTER_27B_200D_DIM = 200
        BERT_VOC = os.path.join(home, 'Downloads/dataset/bert/bert-base-uncased-vocab.txt')
        BERT_MODEL = os.path.join(home, 'Downloads/dataset/bert/bert-base-uncased.tar.gz')

    elif 'home/chang' in home:
        W2V_DIR = os.path.join(home, 'Dropbox/resources/pretrained/word_vec/')
        GLOVE_840B_300D = os.path.join(W2V_DIR, 'glove.840B.300d.txt')
        GLOVE_840B_300D_DIM = 300
        GLOVE_TWITTER_27B_200D = os.path.join(W2V_DIR, 'glove.twitter.27B.200d.txt')
        GLOVE_TWITTER_27B_200D_DIM = 200
        BERT_VOC = os.path.join(home, 'Dropbox/resources/pretrained/bert/bert-base-uncased-vocab.txt')
        BERT_MODEL = os.path.join(home, 'Dropbox/resources/pretrained/bert/bert-base-uncased.tar.gz')
        ROOT_DIR = os.path.join(home, 'Dropbox/project/%s/dataset/%s')
        ROOT_LOCAL_DIR = os.path.join(home, 'project/%s/dataset/%s')
    else:
        W2V_DIR = '/Users/xu052/Documents/project/glove/'
        GLOVE_840B_300D = ''
        BERT_VOC = ''
        BERT_MODEL = ''

    phase_real_str = 'real'
    phase_gan_str = 'gan'
    phase_fake_str = 'fake'


class HP:
    lr = None
    patience = 5
    gen_step = None
    n_epoch_gan = None
    conservative_rate = None
    d_hidden = 768
    dropout = 0.1
    cuda_device = 0
    tune_bert = None
    d_rnn = 128
    d_dense = 128
    filter_width = 5
    num_filters = 128

