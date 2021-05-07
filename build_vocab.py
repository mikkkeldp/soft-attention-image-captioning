import nltk
import pickle
import argparse
from collections import Counter
import itertools
import configparser 

class Flickr8k(object):
    def __init__(self, caption_path):

        with open(caption_path) as f:
            all_captions = f.read().splitlines()

        captions = {}
        for _, idcap in enumerate(all_captions):
            x = idcap.split('#')
            name, cap = x[0], "#".join(x[1:])[2:]
            if name not in captions:
                captions[name] = []
            captions[name].append(cap)

        self.captions = captions


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        # if word not in vocab, return unknown token
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(args):
    """Build a simple vocabulary wrapper."""
    print("Building vocab...")
    counter = Counter()
    f8k_train = Flickr8k(caption_path=args["caption_path"])
    cnt = 0
    for _, cap in f8k_train.captions.items():
        for i in cap:
            tokens = nltk.tokenize.word_tokenize(i.lower())
            counter.update(tokens)
        cnt+=1

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= int(args["threshold"])]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(args):
    vocab = build_vocab(args)
    vocab_path = args["vocab_path"]
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Vocabulary size: ", len(vocab))
    print("Saved the vocabulary to ", vocab_path)


if __name__ == '__main__':

    config = configparser.ConfigParser() 
    config.read("config.ini") 
    args = config["vocab"]

    main(args)
