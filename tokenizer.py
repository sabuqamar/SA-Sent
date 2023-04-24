#!/usr/bin/python
from nltk.tokenize import sent_tokenize, word_tokenize

def tokenize(string):
    sents = sent_tokenize(string)
    ret_l = []
    for sent in sents:
        tokens = word_tokenize(sent)
        ret_l.extend(tokens)
    return ret_l

if __name__ == "__main__":
    tokenize("hello world!")