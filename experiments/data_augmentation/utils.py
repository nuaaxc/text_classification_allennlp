import fasttext
import os
import numpy as np
from pprint import pprint


def train_test_fasttext_classifier(train_path, test_path, lr):
    model = fasttext.train_supervised(train_path,
                                      lr=lr,
                                      epoch=10)
    result_label = model.test_label(test_path)
    result = model.test(test_path)
    f1 = []
    p = []
    r = []
    acc = []
    for label, scores in result_label.items():
        print(label)
        pprint(scores)
        f1.append(scores['f1score'])
        p.append(scores['precision'])
        r.append(scores['recall'])
    print(np.mean(f1), np.mean(p), np.mean(r), result[1])
    # print('saving model ...')
    # model.save_model(path)
    # print('saved.')
