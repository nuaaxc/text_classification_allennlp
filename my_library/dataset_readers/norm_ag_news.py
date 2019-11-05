from config import AGConfig
from my_library.dataset_readers.pre_text_cleaning import clean_dummy_text, clean_normal_text


def normalize_file(input_path, output_path, clean_text):
    with open(input_path, 'r') as f_in, \
            open(output_path, 'w', encoding='utf-8') as f_out:
        print("Reading instances from lines in file at: %s" % input_path)
        for line in f_in.readlines():
            line = line.strip("\n").split('","')
            if not line:
                continue
            label, title, description = line
            text = title + ' ' + description
            label = AGConfig.labels[int(label[1])-1]
            text = clean_text(text)
            f_out.write(label + '\t' + text + '\n')
    print('Saved to %s' % output_path)


if __name__ == '__main__':
    normalize_file(AGConfig.train_raw_path, AGConfig.train_norm_path, clean_normal_text)
    normalize_file(AGConfig.test_raw_path, AGConfig.test_path, clean_normal_text)
    pass



