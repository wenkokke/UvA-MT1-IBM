# coding: utf-8

from collections import defaultdict
from itertools   import chain,product
from msgpack     import pack,unpack
from random      import random
from sys         import stdout
from os          import path


import math
import operator
import numpy as np


class IBM:

    @classmethod
    def load(cls,stream):
        (t,q) = unpack(stream,use_list=False)
        return cls(defaultdict(float,t), defaultdict(float,q))

    def dump(self,stream):
        pack((self.t,self.q), stream)

    def __init__(self, t, q):
        self.t = t
        self.q = q

    def em_train(self,corpus,n=10, s=1):
        for k in range(s,n+s):
            self.em_iter(corpus,passnum=k)
            print("\rPass %2d: 100.00%%" % k)
            print("Likelihood: %.5f" % self.log_likelihood(corpus))

    def log_likelihood(self, corpus):

        likelihood = 0.0
        for k, (f, e) in enumerate(corpus):
            l = len(e) + 1
            m = len(f) + 1
            e = [None] + e
            score = 0.0
            for i in range(1, m):
                score += sum([(self.q[(j, i, l, m)] * self.t[(f[i - 1], e[j])]) / (l ** m)
                          for j in range(0, l)])
            likelihood += math.log(score)

        return likelihood

    def em_iter(self,corpus,passnum=1):

        c1 = defaultdict(float) # ei aligned with fj
        c2 = defaultdict(float) # ei aligned with anything
        c3 = defaultdict(float) # wj aligned with wi
        c4 = defaultdict(float) # wi aligned with anything

        for k, (f, e) in enumerate(corpus):

            if k % 100 == 0:
                stdout.write("\rPass %2d: %6.2f%%" % (passnum, (100*k) / float(len(corpus))))
                stdout.flush()

            l = len(e) + 1
            m = len(f) + 1
            e = [None] + e

            for i in range(1,m):

                num = [ self.q[(j,i,l,m)] * self.t[(f[i - 1], e[j])]
                        for j in range(0,l) ]
                den = float(sum(num))

                for j in range(0,l):

                    delta = num[j] / den

                    c1[(f[i - 1], e[j])] += delta
                    c2[(e[j],)]          += delta
                    c3[(j,i,l,m)]        += delta
                    c4[(i,l,m)]          += delta

        self.t = defaultdict(float,{k: v / c2[k[1:]] for k,v in c1.iteritems() if v > 0.0})
        self.q = defaultdict(float,{k: v / c4[k[1:]] for k,v in c3.iteritems() if v > 0.0})

    def predict_alignment(self,e,f):
        l = len(e) + 1
        m = len(f) + 1
        e = [None] + e

        # for each french word:
        #  - compute a list of indices j of words in the english sentence,
        #    together with the probability of e[j] being aligned with f[i-1]
        #  - take the index j for the word with the _highest_ probability;
        return [
            max(
                [ (j, self.t[(f[i - 1], e[j])] * self.q[(j, i, l, m)])
                  for j in range(0,l) ]
                , key = lambda x: x[1])[0]
            for i in range(1,m) ]


    @classmethod
    def random(cls, corpus):
        return cls.with_generator(
            corpus, lambda n: np.random.dirichlet(np.ones(n), size=1)[0])

    @classmethod
    def uniform(cls,corpus):
        return cls.with_generator(corpus, lambda n: [1 / float(n)] * n)

    @classmethod
    def with_generator(cls,corpus,g):

        # "Compute all possible alignments..."
        lens   = set()
        aligns = defaultdict(set)

        for k, (f, e) in enumerate(corpus):

            stdout.write("\rInit    %6.2f%%" % ((33 * k) / float(len(corpus))))
            stdout.flush()

            e = [None] + e
            lens.add((len(e), len(f) + 1))

            for (f, e) in product(f, e):
                aligns[e].add((f, e))

        # "Compute initial probabilities for each alignment..."
        k = 0
        t = dict()
        for e, aligns_to_e in aligns.iteritems():
            stdout.write("\rInit    %6.2f%%" % (33 + ((33*k) / float(len(aligns)))))
            stdout.flush()
            k += 1

            p_values = g(len(aligns_to_e))
            t.update(zip(aligns_to_e,p_values))

        # "Compute initial probabilities for each distortion..."
        q = dict()
        for k, (l, m) in enumerate(lens):
            stdout.write("\rInit    %6.2f%%" % (66 + ((33*k) / float(len(lens)))))
            stdout.flush()

            for i in range(1,m):
                p_values = g(l)
                for j in range(0,l):
                    q[(j,i,l,m)] = p_values[j]
        print "\rInit     100.00%"

        return cls(t,q)


def read_corpus(path):
    """Read a file as a list of lists of words."""
    with open(path,'r') as f:
        return [ ln.strip().split() for ln in f ]


def main(corpus, ibm, pack_path, corpus_name, n):

    for s in range(1, n + 1):

        curr_pack_path = pack_path + corpus_name + '.' + str(s    ) + '.pack'
        next_pack_path = pack_path + corpus_name + '.' + str(s + 1) + '.pack'

        if path.isfile(curr_pack_path) and not path.isfile(next_pack_path):

            with open(curr_pack_path, 'r') as stream:
                ibm = IBM.load(stream)

            print_test_example(ibm)
            print "Loaded %s" % (curr_pack_path)

        elif not path.isfile(curr_pack_path):

            ibm.em_train(corpus, n=1, s=s)
            print_test_example(ibm)

            with open(curr_pack_path, 'w') as stream:
                ibm.dump(stream)
                print "Dumped %s" % (curr_pack_path)


def print_test_example(ibm):
    e = 'the government is doing what the Canadians want .'.split()
    f = 'le gouvernement fait ce que veulent les Canadiens .'.split()

    a = ibm.predict_alignment(e,f)

    print ' '.join(e)
    print ' '.join(f)
    e = ['NULL'] + e
    print ' '.join([e[j] for j in a])


if __name__ == "__main__":

    # corpus_path = '../data/training/hansards.36.2'
    # fr_corpus_path = corpus_path + '.f'
    # en_corpus_path = corpus_path + '.e'
    #
    # pack_path = corpus_path + '.20.uniform2.pack'
    # with open(pack_path, 'r') as stream:
    #     ibm = IBM.load(stream)
    #     print_test_example(ibm)

    data_path   = 'data'
    corpus_name = '10000'
    corpus_path = path.join(path.dirname(__file__), '..',
                                 data_path, 'training', corpus_name)
    fr_corpus_path   = corpus_path + '.f'
    en_corpus_path   = corpus_path + '.e'
    corpus = zip(read_corpus(fr_corpus_path), read_corpus(en_corpus_path))

    main(corpus, IBM.uniform(corpus), path.join(data_path,'model','ibm2','unif'), corpus_name, 20)
    main(corpus, IBM.random(corpus) , path.join(data_path,'model','ibm2','rand'), corpus_name, 20)
