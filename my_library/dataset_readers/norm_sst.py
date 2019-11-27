from nltk.tree import Tree

from config.sst import SSTCfg
from my_library.dataset_readers.pre_text_cleaning import clean_dummy_text


def normalize_tree_file(input_path, output_path, clean_text):
    with open(input_path, 'r') as f_in, \
            open(output_path, 'w', encoding='utf-8') as f_out:
        print("Reading instances from lines in file at: %s" % input_path)
        for line in f_in.readlines():
            line = line.strip("\n")
            if not line:
                continue
            parsed_line = Tree.fromstring(line)

            text = ' '.join(parsed_line.leaves())
            text = clean_text(text)
            sentiment = parsed_line.label()
            if int(sentiment) == 2:
                continue
            if int(sentiment) < 2:
                label = '__label__negative'
            else:
                label = '__label__positive'

            f_out.write(label + '\t' + text + '\n')
    print('Saved to %s' % output_path)


if __name__ == '__main__':
    normalize_tree_file(SSTCfg.train_raw_path, SSTCfg.train_path, clean_dummy_text)
    normalize_tree_file(SSTCfg.dev_raw_path, SSTCfg.dev_path, clean_dummy_text)
    normalize_tree_file(SSTCfg.test_raw_path, SSTCfg.test_path, clean_dummy_text)
    pass
