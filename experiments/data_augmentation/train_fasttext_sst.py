from experiments.data_augmentation.utils import train_test_fasttext_classifier
from config.sst import SSTCfg


if __name__ == '__main__':
    """
    0.05: 0.6093271033959395 0.6098707282269572 0.6095966504545192 0.6095551894563427
    0.10: 0.6647352192662632 0.6655021189363448 0.6649730762549939 0.6650192202086765
    0.20: 0.712594696969697 0.715753950834065 0.7134303890915408 0.71334431630972
    0.50: 0.7435029475503436 0.7437617702448212 0.7435712321232123 0.7435475013728721
    1.00: 0.8072255108439061 0.8074396413967567 0.8072690492733483 0.8072487644151565
    """
    # train_test_fasttext_classifier(SSTCfg.train_ratio_path % '0.05', SSTCfg.test_path, lr=1)
    # train_test_fasttext_classifier(SSTCfg.train_ratio_path % '0.1', SSTCfg.test_path, lr=1)
    # train_test_fasttext_classifier(SSTCfg.train_ratio_path % '0.2', SSTCfg.test_path, lr=1)
    # train_test_fasttext_classifier(SSTCfg.train_ratio_path % '0.5', SSTCfg.test_path, lr=1)
    # train_test_fasttext_classifier(SSTCfg.train_path, SSTCfg.test_path, lr=0.1)
    pass
