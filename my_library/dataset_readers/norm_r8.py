from pprint import pprint
from config import R8Config


def get_label(input_path):
    labels = set()
    with open(input_path, 'r', encoding='utf-8') as f_in:
        print("Reading instances from lines in file at: %s" % input_path)
        for line in f_in:
            line = line.strip("\n").split('\t')
            if not line or len(line) != 2:
                raise ValueError('Invalid row found. %s' % line)
            label = line[0]
            labels.add(label)
    pprint(labels)


def sanctity_check(input_path):
    with open(input_path, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            line = line.strip().split('\t')
            if len(line) != 2:
                raise Exception(line)


if __name__ == '__main__':
    get_label(R8Config.train_norm_path)
    sanctity_check(R8Config.test_path)





