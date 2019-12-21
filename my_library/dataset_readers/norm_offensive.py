import re

from config.offensive import OffensiveCfg
from my_library.dataset_readers.pre_text_cleaning import clean_dummy_text, clean_tweet_text


def count_vocab_size():
    vocab = set()
    with open(OffensiveCfg.train_raw_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            line = line.strip().split('\t')
            if not line or len(line) != 5:
                raise ValueError('Invalid row found. %s' % line)
            _, text, label, _, _ = line
            text = clean_tweet_text(text, remove_stop=False)
            vocab.update(text.split())

    with open(OffensiveCfg.test_raw_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            line = line.strip().split('\t')
            if not line or len(line) != 2:
                raise ValueError('Invalid row found. %s' % line)
            _, text = line
            text = clean_tweet_text(text, remove_stop=False)
            vocab.update(text.split())
    print(len(vocab))


def normalize_train_file(input_path, output_path, clean_text, skip_header):
    with open(input_path, 'r', encoding='utf-8') as f_in, \
            open(output_path, 'w', encoding='utf-8') as f_out:
        print("Reading instances from lines in file at: %s" % input_path)
        if skip_header:
            next(f_in)
        for line in f_in:
            line = line.strip().split('\t')
            if not line or len(line) != 5:
                raise ValueError('Invalid row found. %s' % line)
            _, text, label, _, _ = line
            text = re.sub(r'@USER', '', text)  # remove @USER
            text = re.sub(r'URL', '', text)  # remove URL
            text = clean_text(text, remove_stop=False)
            if len(text) > 0:
                f_out.write(label + '\t' + text + '\n')
            else:
                print(line[1])
    print('Saved to %s' % output_path)


def normalize_test_file(input_path, output_path, clean_text, skip_header):
    id_label = {}
    with open(OffensiveCfg.ground_truth_path) as f:
        for line in f:
            _id, label = line.strip().split(',')
            id_label[_id] = label

    with open(input_path, 'r', encoding='utf-8') as f_in, \
            open(output_path, 'w', encoding='utf-8') as f_out:
        print("Reading instances from lines in file at: %s" % input_path)
        if skip_header:
            next(f_in)
        for line in f_in:
            line = line.strip().split('\t')
            if not line or len(line) != 2:
                raise ValueError('Invalid row found. %s' % line)
            _id, text, = line
            label = id_label[_id]
            text = re.sub(r'@USER', '', text)  # remove USER_user
            text = re.sub(r'URL', '', text)  # remove URL
            text = clean_text(text, remove_stop=False)
            if len(text) > 0:
                f_out.write(label + '\t' + text + '\n')
            else:
                print(line[1])
    print('Saved to %s' % output_path)


if __name__ == '__main__':
    normalize_train_file(OffensiveCfg.train_raw_path,
                         OffensiveCfg.train_norm_path,
                         clean_tweet_text,
                         skip_header=True)
    normalize_test_file(OffensiveCfg.test_raw_path,
                        OffensiveCfg.test_path,
                        clean_tweet_text,
                        skip_header=True)
    # count_vocab_size()
    pass



