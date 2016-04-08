import math
import sys
import numpy as np


class SentencePair:
    def __init__(self, french, english):
        self.french = french
        self.english = english
        self.m = len(self.french)
        self.l = len(self.english)
        self.a = []

class Vocabulary:
    def __init__(self):
        self.tokens = []
        self.token_map = {}

    def visit_token(self, token):
        if token not in self.token_map:
            self.token_map[token] = len(self.tokens)
            self.tokens.append(token)
        return self.token_map[token]

    def __getitem__(self, i):
        return self.tokens[i]

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        return iter(self.tokens)

    def __contains__(self, key):
        return key in self.token_map


class Corpus:
    def __init__(self, french_filename, english_filename):
        i = 0
        french_file_pointer = open(french_filename, 'r')
        english_file_pointer = open(english_filename, 'r')
        self.french_vocabulary = Vocabulary()
        self.english_vocabulary = Vocabulary()
        self.sentence_pairs = []

        while True:
            try:
                french_line = french_file_pointer.next()
                english_line = english_file_pointer.next()
            except StopIteration:
                break

            french_tokens = french_line.split()
            french_sentence = []
            for token in french_tokens:
                french_sentence.append(
                    self.french_vocabulary.visit_token(token)
                )

            english_tokens = english_line.split()
            english_sentence = []
            for token in english_tokens:
                english_sentence.append(
                    self.english_vocabulary.visit_token(token)
                )

                i += 1
                if i % 10000 == 0:
                    sys.stdout.flush()
                    sys.stdout.write("\rReading corpus: %d" % i)

            sentence_pair = SentencePair(french_sentence, english_sentence)
            self.sentence_pairs.append(sentence_pair)


        sys.stdout.flush()
        print "\rCorpus read: %d" % i

        french_file_pointer.close()
        english_file_pointer.close()

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, i):
        return self.sentence_pairs[i]

class Model:

    def __init__(self, corpus):
        self.corpus = corpus
        # c(e,f) Key is tuple where first value is english token, second value french token
        self.token_tuple_count_map = {}
        # c(e) Key is english token
        self.token_count_map = {}
        # Key is tuple (triple) c(j|i,m,l) where i = 1 .. m, j = 0 .. l
        self.alignment_count_map = {}
        # Key is tuple (triple) c(i,m,l) where i = 1 .. m
        self.alignment_start_count_map = {}

    def run(self):
        for sentence_pair in self.corpus:
            for i in range(1, sentence_pair.m):
                for j in range(0, sentence_pair.l):
                    m = sentence_pair.m
                    l = sentence_pair.l
                    fi = sentence_pair.french[i]
                    ej = sentence_pair.english[j]
                    self.token_tuple_count_map[(ej,fi)] = self.token_tuple_count_map[(ej,fi)] + 1
                    self.token_count_map[ej] = self.token_count_map[ej] + 1
                    self.alignment_count_map[(j,i,m,l)] = self.alignment_count_map[(j,i,m,l)] + 1
                    self.alignment_start_count_map[(i, m, l)] = self.alignment_count_map[(j, i, m, l)] + 1

if __name__ == '__main__':

    corpus = Corpus('../data/training/hansards.36.2.f', '../data/training/hansards.36.2.e')

    model = Model(corpus)

