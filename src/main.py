import math
import sys
import random
import numpy as np
import pickle
import os.path

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
            self.token_map[token] = len(self.tokens) + 1
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
        self.max_m = 0
        self.max_l = 0

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

            if self.max_l < sentence_pair.l:
                self.max_l = sentence_pair.l

            if self.max_m < sentence_pair.m:
                self.max_m = sentence_pair.m

            self.sentence_pairs.append(sentence_pair)


        sys.stdout.flush()
        print "\rCorpus read: %d" % i

        french_file_pointer.close()
        english_file_pointer.close()

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, i):
        return self.sentence_pairs[i]

    def __iter__(self):
        return iter(self.sentence_pairs)


class Model:

    def __init__(self, corpus):
        self.corpus = corpus

        self.token_tuple_count_map = {}
        self.token_count_map = {}
        self.alignment_count_map = {}
        self.alignment_start_count_map = {}

        # Key is tuple (triple) q(j|i,m,l) where i = 1 .. m, j = 0 .. l
        self.q = {}
        # t(e,f) Key is tuple where first value is english token, second value french token
        self.t = {}

    def run(self, i):

        # self.random_init()

        sys.stdout.flush()
        sys.stdout.write("\rIteration: 0")

        for i in range(0, i):
            self.e_step()
            self.m_step()

            sys.stdout.flush()
            sys.stdout.write("\rIteration: %d" % i)

        sys.stdout.flush()
        print "\rEM finished"

    def e_step(self):

        # c(e,f) Key is tuple where first value is english token, second value french token
        self.token_tuple_count_map = {}
        # c(e) Key is english token
        self.token_count_map = {}
        # Key is tuple (triple) c(j|i,m,l) where i = 1 .. m, j = 0 .. l
        self.alignment_count_map = {}
        # Key is tuple (triple) c(i,m,l) where i = 1 .. m
        self.alignment_start_count_map = {}

        for k in self.corpus:
            for i in range(1, k.m + 1):
                for j in range(0, k.l + 1):
                    fi = k.french[i - 1]

                    if j == 0:
                        ej = 0 # NULL word
                    else:
                        ej = k.english[j - 1]

                    if (ej, fi) in self.token_tuple_count_map:
                        self.token_tuple_count_map[(ej, fi)] += self.d(k, i, j)
                    else:
                        self.token_tuple_count_map[(ej, fi)] = self.d(k, i, j)

                    if ej in self.token_count_map:
                        self.token_count_map[ej] += self.d(k, i, j)
                    else:
                        self.token_count_map[ej] = self.d(k, i, j)

                    if (j, i, k.m, k.l) in self.alignment_count_map:
                        self.alignment_count_map[(j, i, k.m, k.l)] += self.d(k, i, j)
                    else:
                        self.alignment_count_map[(j, i, k.m, k.l)] = self.d(k, i, j)

                    if (i, k.m, k.l) in self.alignment_start_count_map:
                        self.alignment_start_count_map[(i, k.m, k.l)] += self.d(k, i, j)
                    else:
                        self.alignment_start_count_map[(i, k.m, k.l)] = self.d(k, i, j)

    def m_step(self):

        for key in self.q:
            self.q[key] = self.alignment_count_map[key] / self.alignment_start_count_map[(key[1], key[2], key[3])]

        for key in self.t:
            self.t[key] = self.token_tuple_count_map[key] / self.token_count_map[key[0]]

    def d(self, k, i, j):
        fi = k.french[i - 1]
        the_sum = 0

        for jj in range(0, k.l + 1):
            if jj == 0:
                ejj = 0  # NULL word
            else:
                ejj = k.english[jj - 1]

            if not (ejj, fi) in self.t:
                self.t[(ejj, fi)] = random.random()

            if not (jj, i, k.m, k.l) in self.q:
                self.q[(jj, i, k.m, k.l)] = random.random()

            the_sum += self.q[(jj, i, k.m, k.l)] * self.t[(ejj, fi)]

        if j == 0:
            ej = 0 # NULL word
        else:
            ej = k.english[j - 1]

        return self.q[(j, i, k.m, k.l)] * self.t[(ej, fi)] / the_sum

if __name__ == '__main__':

    if os.path.isfile('../tmp/model'):
        model = pickle.load(open('../tmp/model', 'r'))

    if os.path.isfile('../tmp/corpus.1'):
        corpus = pickle.load(open('../tmp/corpus.1', 'r'))
    else:
        corpus = Corpus('../data/training/hansards.36.2.f.1', '../data/training/hansards.36.2.e.1')
        pickle.dump(corpus, open('../tmp/corpus.1', 'w'))

    model = Model(corpus)
    model.run(10)
    pickle.dump(model, open('../tmp/model', 'w'))
