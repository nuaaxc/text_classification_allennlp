import os
import random
from collections import defaultdict

from config import StanceConfig, YahooConfig, TRECConfig
from my_library.dataset_readers.clearn_text import clean_tweet_text, clean_normal_text


def train_dev_split(train_raw_path,
                    train_ratio_path,
                    dev_ratio_path,
                    test_raw_path,
                    test_path,
                    encoding,
                    sep,
                    label_index,
                    text_index,
                    skip_header,
                    clean_text,
                    sample_ratio=0.1,
                    split_ratio=0.9):
    random.seed(2019)
    label_example = defaultdict(list)
    with open(train_raw_path, encoding=encoding) as f:
        if skip_header:
            next(f)
        for line in f:
            label = line.strip().split(sep)[label_index]
            text = line.strip().split(sep)[text_index]
            label_example[label].append(text)

    label_train_sample = defaultdict()
    label_dev_sample = defaultdict()
    for label, example_list in label_example.items():
        print(label)
        training_size = len(example_list)
        print('\ttraining size:', training_size)
        sample_size = int(sample_ratio * training_size)
        print('\tsample size:', sample_size)
        samples = random.sample(example_list, sample_size)
        random.shuffle(samples)
        label_train_sample[label] = samples[:int(len(samples) * split_ratio)]
        label_dev_sample[label] = samples[int(len(samples) * split_ratio):]

    print('Sampled training set:')
    for label, example_list in label_train_sample.items():
        print(label, len(example_list))
    print('Sampled dev set:')
    for label, example_list in label_dev_sample.items():
        print(label, len(example_list))

    # Check
    for label in label_example.keys():
        all_examples = label_example[label]
        sampled_train = label_train_sample[label]
        sampled_dev = label_dev_sample[label]
        print('[Label]', label)
        assert len(sampled_train)
        assert len(sampled_dev)
        print('training set size:', len(sampled_train))
        print('development set size:', len(sampled_dev))
        print('train-dev duplicates:', len(set(sampled_train) & set(sampled_dev)))
        assert len(set(sampled_train) & set(all_examples)) == len(set(sampled_train))
        assert len(set(sampled_dev) & set(all_examples)) == len(set(sampled_dev))

    """
    Saving to file
    """
    print('[saving] sampled training set ...')
    with open(train_ratio_path, 'w', encoding='utf-8') as f:
        for label, samples in label_train_sample.items():
            for sample in samples:
                sample = clean_text(sample)
                # assert len(sample) > 0
                if len(sample) > 0:
                    f.write(label + '\t' + sample)
                    f.write('\n')

    print('[saving] sampled dev set ...')
    with open(dev_ratio_path, 'w', encoding='utf-8') as f:
        for label, samples in label_dev_sample.items():
            for sample in samples:
                sample = clean_text(sample)
                # assert len(sample) > 0
                if len(sample) > 0:
                    f.write(label + '\t' + sample)
                    f.write('\n')

    print('[saving] test set ...')
    with open(test_raw_path, 'r', encoding='utf-8') as f_test_raw, \
            open(test_path, 'w', encoding='utf-8') as f_test:
        if skip_header:
            next(f_test_raw)
        for line in f_test_raw:
            label = line.strip().split(sep)[label_index]
            text = line.strip().split(sep)[text_index]
            text = clean_text(text)
            # assert len(text) > 0
            if len(text) > 0:
                f_test.write(label + '\t' + text)
                f_test.write('\n')
    print('[saved]')


def dataset_stance():
    sample_ratio = 0.3
    stance_target = 'a'
    train_dev_split(train_raw_path=StanceConfig.train_raw_path % stance_target,
                    train_ratio_path=StanceConfig.train_ratio_path % (stance_target, int(sample_ratio * 100)),
                    dev_ratio_path=StanceConfig.dev_ratio_path % (stance_target, int(sample_ratio * 100)),
                    test_raw_path=StanceConfig.test_raw_path % stance_target,
                    test_path=StanceConfig.test_path % stance_target,
                    encoding='windows-1251',
                    sep='\t',
                    label_index=3,
                    text_index=2,
                    skip_header=True,
                    clean_text=clean_tweet_text,
                    sample_ratio=sample_ratio)


def dataset_TREC(sample_ratio=0.2):

    train_dev_split(train_raw_path=TRECConfig.train_norm_path,
                    train_ratio_path=TRECConfig.train_ratio_path % int(sample_ratio * 100),
                    dev_ratio_path=TRECConfig.dev_ratio_path % int(sample_ratio * 100),
                    test_raw_path=TRECConfig.test_norm_path,
                    test_path=TRECConfig.test_path % '100',
                    encoding='windows-1251',
                    sep='\t',
                    label_index=0,
                    text_index=1,
                    skip_header=False,
                    clean_text=clean_normal_text,
                    sample_ratio=sample_ratio)


if __name__ == '__main__':
    dataset_TREC()
