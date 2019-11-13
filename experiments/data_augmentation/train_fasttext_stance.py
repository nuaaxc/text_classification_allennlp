from experiments.data_augmentation.utils import train_test_fasttext_classifier
from config import StanceConfig


if __name__ == '__main__':
    """
    0.05: 0.27049317958192637 0.26715306277549783 0.36253167122732344 0.36429143314651724
    0.10: 0.40131223268709215 0.42163810151615033 0.4119215833746726 0.500400320256205
    0.20: 0.41346423020470585 0.42089728612462407 0.41143431320776863 0.510808646917534
    0.50: 0.4958032089329832 0.4970427828332212 0.49477471422551506 0.567654123298639
    1.00: 0.5429696781005843 0.5378942125510894 0.5549416449302033 0.5916733386709367
    """
    # train_test_fasttext_classifier(StanceConfig.train_ratio_path % '0.05', StanceConfig.test_path, lr=1)
    # train_test_fasttext_classifier(StanceConfig.train_ratio_path % '0.1', StanceConfig.test_path, lr=1)
    # train_test_fasttext_classifier(StanceConfig.train_ratio_path % '0.2', StanceConfig.test_path, lr=1)
    # train_test_fasttext_classifier(StanceConfig.train_ratio_path % '0.5', StanceConfig.test_path, lr=1)
    # train_test_fasttext_classifier(StanceConfig.train_norm_path, StanceConfig.test_path, lr=0.3)
    pass
