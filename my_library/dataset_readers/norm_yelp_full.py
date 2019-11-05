from config import YelpFullConfig
from my_library.dataset_readers.pre_text_cleaning import clean_dummy_text, clean_normal_text


def normalize_file(input_path, output_path, clean_text):
    with open(input_path, 'r') as f_in, \
            open(output_path, 'w', encoding='utf-8') as f_out:
        print("Reading instances from lines in file at: %s" % input_path)
        c = 0
        for line in f_in.readlines():
            line = line.strip("\n").split('","')
            if not line or len(line) > 2:
                print(line)
                continue
            label, text = line
            label = label[-1]
            text = clean_text(text)
            f_out.write(label + '\t' + text + '\n')
            c += 1
            if c % 10000 == 0:
                print(c)
    print('Saved to %s' % output_path)


if __name__ == '__main__':
    normalize_file(YelpFullConfig.train_raw_path, YelpFullConfig.train_norm_path, clean_normal_text)
    # normalize_file(YelpFullConfig.test_raw_path, YelpFullConfig.test_path, clean_normal_text)
    pass



