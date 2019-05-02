import random
from collections import defaultdict

from config import DBPediaConfig, YahooConfig
from my_library.dataset_readers.clearn_text import clean_normal_text


def train_dev_split(config, sample_ratio=0.1, split_ratio=0.9):
    random.seed(2019)
    label_example = defaultdict(list)
    with open(config.train_path, encoding='utf-8') as f:
        for line in f:
            label = line.strip().split(',"')[0]
            label_example[label].append(line)

    training_size = None
    print('Training size for each class:')
    for label, example_list in label_example.items():
        training_size = len(example_list)
        print(label, training_size)

    sample_size = int(sample_ratio * training_size)
    print('Sample size:', sample_size)

    label_train_sample = defaultdict()
    label_dev_sample = defaultdict()
    for label, example_list in label_example.items():
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
        assert len(set(sampled_train) & set(sampled_dev)) == 0
        assert len(set(sampled_train) & set(all_examples)) == len(set(sampled_train))
        assert len(set(sampled_dev) & set(all_examples)) == len(set(sampled_dev))

    # save to file
    print('saving sampled training set ...')
    with open(config.train_ratio_path % int(sample_ratio*100), 'w', encoding='utf-8') as f:
        for label, samples in label_train_sample.items():
            for sample in samples:
                sample = clean_normal_text(sample)
                f.write(sample)
                f.write('\n')

    print('saving sampled dev set ...')
    with open(config.dev_ratio_path % int(sample_ratio * 100), 'w', encoding='utf-8') as f:
        for label, samples in label_dev_sample.items():
            for sample in samples:
                sample = clean_normal_text(sample)
                f.write(sample)
                f.write('\n')
    print('saved.')


if __name__ == '__main__':
    train_dev_split(YahooConfig, sample_ratio=0.01)

