import re
import spacy
spacy_en = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])


def clean_dummy_text(string):
    return string


def clean_tweet_text(tweet, remove_stop):
    tweet = tweet.lower()  # lower case
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)  # URLs
    tweet = re.sub(r'@(\S+)', r'USER_\1', tweet)  # @handle
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)  # #hashtag
    tweet = re.sub(r'\brt\b', '', tweet)  # Remove RT
    tweet = re.sub(r'\.{2,}', ' ', tweet)  # Replace 2+ dots with space
    tweet = tweet.strip(' "\'')  # Strip space, " and ' from tweet
    tweet = preprocess_emojis(tweet)  # Replace emojis with either EMO_POS or EMO_NEG
    tweet = re.sub(r'\s+', ' ', tweet)  # Replace multiple spaces with a single space

    processed_tweet = []
    for word in tweet.split():
        word = preprocess_word(word)
        if remove_stop:
            _stopwords = stopwords
        else:
            _stopwords = set()
        if is_valid_word(word) and word not in _stopwords:
            processed_tweet.append(word)

    tweet = ' '.join(processed_tweet)
    # tokens = [tok.lemma_ for tok in spacy_en.tokenizer(tweet) if tok.lemma]
    tokens = [tok.text for tok in spacy_en(tweet)]
    return ' '.join(tokens)


def clean_normal_text(string):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    string = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', string)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    string = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', string)
    # Love -- <3, :*
    string = re.sub(r'(<3|:\*)', ' EMO_POS ', string)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    string = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', string)
    # Sad -- :-(, : (, :(, ):, )-:
    string = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', string)
    # Cry -- :,(, :'(, :"(
    string = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', string)

    string = string.lower()
    string = re.sub(r'\\n', ' ', string)
    string = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', string)  # URLs
    string = re.sub(r'\S*@\S*\s?', '', string)  # email
    string = re.sub(r'(?<!\w)([a-z])\.', r'\1', string)  # remove periods in acronyms
    string = re.sub(r'(?<!\w)([a-z])', r'\1', string)  # remove periods in acronyms
    # string = re.sub(r'e\.t\.r\.', 'etr', string)
    # string = re.sub(r"u\.s\.", " us ", string)
    # string = re.sub(r"u\.s\.a\.", " usa ", string)
    # string = re.sub(r"u\.s\.a", " usa ", string)
    # string = re.sub(r"e\.g\.,", " ", string)
    # string = re.sub(r"a\.k\.a\.", " ", string)
    # string = re.sub(r"i\.e\.,", " ", string)
    # string = re.sub(r"i\.e\.", " ", string)
    string = re.sub(r"[^A-Za-z0-9,.!? ]", "", string)
    string = re.sub(r'(.)\1+', r'\1\1', string)  # remove duplicate character
    string = re.sub(r'([a-z0-9]+(-[a-z0-9]+)*\.)+[a-z]{2,}', '', string)
    string = re.sub(r",", " SEP ", string)
    string = re.sub(r"br", "", string)
    string = re.sub(r"!", " SEP ", string)
    string = re.sub(r"\?", " SEP ", string)
    string = re.sub(r"\.", " SEP ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r'[-|\']', '', string)
    string = re.sub(r'\b(\w+)( \1\b)+', r'\1', string)  # remove repeated words
    # string = re.sub(r'SEP', '[SEP]', string)
    string = re.sub(r'SEP', '.', string)

    return string


def is_valid_word(word):
    return re.search(r'^[a-zA-Z][a-z0-9A-Z._]*$', word) is not None  # Check if word begins with an alphabet


def preprocess_word(word):
    word = word.strip('“”\'"?!,.():;‘’—_*')  # Remove punctuation
    word = re.sub(r'(.)\1+', r'\1\1', word)  # Convert >= 2 letter repetitions to 2 letter, e.g., funnnnny --> funny
    word = re.sub(r'[-|\']', '', word)  # Remove - & '
    word = re.sub(
        r"""(?:^(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)$)|n't""",
        'not', word)
    return word


def preprocess_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet


stopwords = set(["semst", "\"", "#", "$", "%", "&", "\\", "'", "(", ")",
                 "*", ",", "-", ".", "/", ":", ";", "<", ">", "@",
                 "[", "]", "^", "_", "`", "{", "|", "}", "~", "=", "+", "!", "?"] + \
                ["a", "about", "above", "after", "again", "am", "an", "and", "any", "are", "as", "at",
                 "be",
                 "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
                 "does",
                 "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
                 "he",
                 "he'd",
                 "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
                 "i",
                 "i'd",
                 "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more",
                 "most",
                 "my", "myself", "nor", "of", "on", "once", "or", "other", "ought", "our", "ours", "ourselves",
                 "out",
                 "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than",
                 "that",
                 "that's",
                 "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd",
                 "they'll",
                 "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
                 "we",
                 "we'd", "will",
                 "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which",
                 "while", "who",
                 "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your",
                 "yours",
                 "yourself", "yourselves"])
