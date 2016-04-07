import math
import sys
import numpy as np


class SentencePair:
    def __init__(self, english, french):
        self.english = english
        self.french = french

class Vocabulary:
    def __init__(self):
        self.tokens = []
        self.token_map = {}

    def visit_token(self, token):
        if token not in self.token_map:
            self.token_map[token] = len(self.tokens)
            self.tokens.append(token)

    def __getitem__(self, i):
        return self.tokens[i]

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        return iter(self.tokens)

    def __contains__(self, key):
        return key in self.token_map


class Sentence:
    def __init__(self, sentence):
        self.sentence = sentence


class Corpus:
    def __init__(self, french_filename, english_filename):
        i = 0
        french_file_pointer = open(french_filename, 'r')
        english_file_pointer = open(english_filename, 'r')
        french_vocabulary = Vocabulary()
        english_vocabulary = Vocabulary()

        while True:
            try:
                french_line = french_file_pointer.next()
                english_line = english_file_pointer.next()
            except StopIteration:
                break

            french_tokens = french_line.split()
            for token in french_tokens:
                french_vocabulary.visit_token(token)

            english_tokens = english_line.split()
            for token in english_tokens:
                english_vocabulary.visit_token(token)

                i += 1
                if i % 10000 == 0:
                    sys.stdout.flush()
                    sys.stdout.write("\rReading corpus: %d" % i)

        sys.stdout.flush()
        print "\rCorpus read: %d" % i

        french_file_pointer.close()
        english_file_pointer.close()


if __name__ == '__main__':

    Corpus('../data/training/hansards.36.2.f', '../data/training/hansards.36.2.e')
