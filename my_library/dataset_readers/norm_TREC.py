from config.trec import TRECCfg


def norm(in_file_path, out_file_path):
    with open(in_file_path, 'r', encoding='windows-1251') as f_test_raw, \
            open(out_file_path, 'w', encoding='utf-8') as f_test_norm:
        for line in f_test_raw:
            line = line.strip().split(' ')
            label = line[0].split(':')[0]
            text = line[1:]
            f_test_norm.write(label + '\t' + ' '.join(text) + '\n')


if __name__ == '__main__':
    norm(TRECCfg.train_raw_path, TRECCfg.train_norm_path)
    norm(TRECCfg.test_raw_path, TRECCfg.test_path)
