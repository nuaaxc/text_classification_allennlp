from config import AffectConfig
from my_library.dataset_readers.pre_text_cleaning import clean_dummy_text, clean_tweet_text


def normalize_file(input_path, output_path, clean_text, skip_header):
    with open(input_path, 'r', encoding='utf-8') as f_in, \
            open(output_path, 'w', encoding='utf-8') as f_out:
        print("Reading instances from lines in file at: %s" % input_path)
        if skip_header:
            next(f_in)
        for line in f_in:
            line = line.strip("\n").split('\t')
            if not line or len(line) != 4:
                raise ValueError('Invalid row found. %s' % line)
            label = int(line[-1].split(':')[0])
            text = line[1]
            text = clean_text(text)
            f_out.write(str(label) + '\t' + text + '\n')
    print('Saved to %s' % output_path)


if __name__ == '__main__':
    normalize_file(AffectConfig.train_raw_path, AffectConfig.train_path, clean_tweet_text, True)
    normalize_file(AffectConfig.dev_raw_path, AffectConfig.dev_path, clean_tweet_text, True)
    normalize_file(AffectConfig.test_raw_path, AffectConfig.test_path, clean_tweet_text, True)
    pass



