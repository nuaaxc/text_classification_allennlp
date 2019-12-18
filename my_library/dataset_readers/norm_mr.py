from config.mr import MRCfg

from sklearn.model_selection import train_test_split
from my_library.dataset_readers.pre_text_cleaning import clean_dummy_text, clean_normal_text


def normalize_file(clean_text):
    X, y = [], []

    with open(MRCfg.raw_pos, 'r') as f:
        for line in f:
            line = clean_text(line.strip())
            X.append(line)
            y.append('__label__POS')

    with open(MRCfg.raw_neg, 'r') as f:
        for line in f:
            line = clean_text(line.strip())
            X.append(line)
            y.append('__label__NEG')

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.1, random_state=2020)

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    print(len(X_train), len(y_train))
    print(len(X_test), len(y_test))

    with open(MRCfg.train_norm_path, 'w', encoding='utf-8') as f:
        for i in range(len(X_train)):
            if len(X_train[i]) > 0:
                f.write(y_train[i] + '\t' + X_train[i] + '\n')

    with open(MRCfg.test_path, 'w', encoding='utf-8') as f:
        for i in range(len(X_test)):
            if len(X_test[i]) > 0:
                f.write(y_test[i] + '\t' + X_test[i] + '\n')


if __name__ == '__main__':
    normalize_file(clean_normal_text)
