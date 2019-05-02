import re


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
    string = re.sub(r'\S*@\S*\s?', '', string)                  # email
    string = re.sub(r'(?<!\w)([a-z])\.', r'\1', string)         # remove periods in acronyms
    string = re.sub(r'(?<!\w)([a-z])', r'\1', string)           # remove periods in acronyms
    # string = re.sub(r'e\.t\.r\.', 'etr', string)
    # string = re.sub(r"u\.s\.", " us ", string)
    # string = re.sub(r"u\.s\.a\.", " usa ", string)
    # string = re.sub(r"u\.s\.a", " usa ", string)
    # string = re.sub(r"e\.g\.,", " ", string)
    # string = re.sub(r"a\.k\.a\.", " ", string)
    # string = re.sub(r"i\.e\.,", " ", string)
    # string = re.sub(r"i\.e\.", " ", string)
    string = re.sub(r"[^A-Za-z0-9,.!? ]", "", string)
    string = re.sub(r'(.)\1+', r'\1\1', string)                 # remove duplicate character
    string = re.sub(r'([a-z0-9]+(-[a-z0-9]+)*\.)+[a-z]{2,}', '', string)
    string = re.sub(r",", " SEP ", string)
    string = re.sub(r"br", "", string)
    string = re.sub(r"!", " SEP ", string)
    string = re.sub(r"\?", " SEP ", string)
    string = re.sub(r"\.", " SEP ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r'[-|\']', '', string)
    string = re.sub(r'\b(\w+)( \1\b)+', r'\1', string)          # remove repeated words
    string = re.sub(r'SEP', '[SEP]', string)
    return string.strip()


def is_valid_word(word):
    return re.search(r'^[a-zA-Z][a-z0-9A-Z._]*$', word) is not None  # Check if word begins with an alphabet


def preprocess_word(word):
    word = word.strip('“”\'"?!,.():;‘’')            # Remove punctuation
    word = re.sub(r'(.)\1+', r'\1\1', word)     # Convert >= 2 letter repetitions to 2 letter, e.g., funnnnny --> funny
    word = re.sub(r'[-|\']', '', word)          # Remove - & '
    word = re.sub(r"""(?:^(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)$)|n't""", 'NEGATION', word)
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


def preprocess(tweet, remove=True):
    processed_tweet = []
    tweet = tweet.lower()  # lower case
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)  # URLs
    if remove:
        tweet = re.sub(r'@(\S+)', '', tweet)  # remove @handle
        tweet = re.sub(r'#(\S+)', '', tweet)  # remove #hashtag
    else:
        tweet = re.sub(r'@(\S+)', r'USER_\1', tweet)  # @handle
        tweet = re.sub(r'#(\S+)', r' \1 ', tweet)  # #hashtag
    tweet = re.sub(r'\brt\b', '', tweet)    # Remove RT
    tweet = re.sub(r'\.{2,}', ' ', tweet)   # Replace 2+ dots with space
    if remove:
        tweet = re.sub(r'adani', '', tweet)     # remove adani
        tweet = re.sub(r'bhp', '', tweet)       # remove bhp
        tweet = re.sub(r'santos', '', tweet)  # remove santos
        tweet = re.sub(r'rioTinto', '', tweet)  # remove rioTinto
        tweet = re.sub(r'fortescue', '', tweet)  # remove fortescue
    tweet = tweet.strip(' "\'')  # Strip space, " and ' from tweet
    tweet = preprocess_emojis(tweet)  # Replace emojis with either EMO_POS or EMO_NEG
    tweet = re.sub(r'\s+', ' ', tweet)  # Replace multiple spaces with a single space

    words = tweet.split()
    for word in words:
        word = preprocess_word(word)
        if is_valid_word(word):
            processed_tweet.append(word)

    return ' '.join(processed_tweet)