from nltk.tree import Tree

from config import SSTConfig
from my_library.dataset_readers.pre_text_cleaning import clean_dump_text


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
                label = 'negative'
            else:
                label = 'positive'

            f_out.write(label + '\t' + text + '\n')
    print('Saved to %s' % output_path)


if __name__ == '__main__':
    normalize_tree_file(SSTConfig.train_raw_path, SSTConfig.train_path, clean_dump_text)
    normalize_tree_file(SSTConfig.dev_raw_path, SSTConfig.dev_path, clean_dump_text)
    normalize_tree_file(SSTConfig.test_raw_path, SSTConfig.test_path, clean_dump_text)
    pass
