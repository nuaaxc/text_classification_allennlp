import logging

from config import StanceConfig

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def baby_mention_stats():
    filename = 'C:/Users/nuaax/OneDrive/data61/project/stance_classification/dataset/semeval/data/la-all.txt'
    labels = []
    for line in open(filename, encoding='utf-8'):
        text = line.strip().split('\t')[2]
        label = line.strip().split('\t')[3]
        text = text.lower()
        if 'baby' in text:
            labels.append(label)
    import collections
    c = collections.Counter(labels)
    print(c)


def merge_all_stance_target(input_file_path, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        f_out.write('ID\tTarget\tTweet\tStance\n')
        for stance_target in StanceConfig.target:
            print('processing %s ...' % stance_target)
            with open(input_file_path % stance_target, encoding='windows-1251') as f_in:
                next(f_in)
                for line in f_in:
                    f_out.write(line)


if __name__ == '__main__':
    # baby_mention_stats()
    # merge_all_stance_target(StanceConfig.train_raw_path, StanceConfig.train_raw_path_all)
    # merge_all_stance_target(StanceConfig.test_raw_path, StanceConfig.test_raw_path_all)
    pass



