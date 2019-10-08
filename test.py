



from config import TRECConfig



with open(TRECConfig.train_raw_path) as f:
    s = set()
    for line in f:
        label = line.strip().split(':')[0]
        s.add(label)
    print(s)
