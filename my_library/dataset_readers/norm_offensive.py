from config import OffensiveConfig
from my_library.dataset_readers.pre_text_cleaning import clean_dummy_text, clean_tweet_text


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
            text = clean_text(text, remove_stop=False)
            if len(text) > 0:
                f_out.write(label + '\t' + text + '\n')
            else:
                raise ValueError('Empty text after pre-processing. %s' % text)
    print('Saved to %s' % output_path)


def normalize_test_file(input_path, output_path, clean_text, skip_header):
    id_label = {}
    with open(OffensiveConfig.ground_truth_path) as f:
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
            text = clean_text(text, remove_stop=False)
            if len(text) > 0:
                f_out.write(label + '\t' + text + '\n')
            else:
                raise ValueError('Empty text after pre-processing. %s' % text)
    print('Saved to %s' % output_path)


if __name__ == '__main__':
    # normalize_train_file(OffensiveConfig.train_raw_path,
    #                      OffensiveConfig.train_norm_path,
    #                      clean_tweet_text,
    #                      skip_header=True)
    # normalize_test_file(OffensiveConfig.test_raw_path,
    #                     OffensiveConfig.test_path,
    #                     clean_tweet_text,
    #                     skip_header=True)
    pass



