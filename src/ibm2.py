# coding: utf-8
import os
from collections import defaultdict
from itertools   import product

import operator
from msgpack     import pack,unpack
from sys         import stdout

import math
import numpy as np
import time


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


    def log_likelihood(self, corpus):

        start = time.time()

        likelihood = 0.0
        for k, (f, e) in enumerate(corpus):

            if k % 1000 == 0:
                stdout.write("\rLog-likelihood calculations: %6.2f%%" % ((100 * k) / float(len(corpus))))
                stdout.flush()

            l = len(e) + 1
            m = len(f) + 1
            e = [None] + e

            score = 0.0
            for i in range(1, m):
                score += sum(
                    [ (self.q[(j, i, l, m)] * self.t[(f[i - 1], e[j])]) / (m * l)
                      for j in range(0, l)])
            likelihood += math.log(score)

        print("\rLog-likelihood: %.5f (Elapsed: %.2fs)" % (likelihood,(time.time() - start)))

        return likelihood


    def em_iter(self,corpus,passnum=1):

        start = time.time()

        c1 = defaultdict(float) # ei aligned with fj
        c2 = defaultdict(float) # ei aligned with anything
        c3 = defaultdict(float) # wj aligned with wi
        c4 = defaultdict(float) # wi aligned with anything

        likelihood = 0.0

        for k, (f, e) in enumerate(corpus):

            if k % 1000 == 0:
                stdout.write("\rPass %2d: %6.2f%%" % (passnum, (100*k) / float(len(corpus))))
                stdout.flush()

            l = len(e) + 1
            m = len(f) + 1
            e = [None] + e

            for i in range(1,m):

                num = [ self.q[(j,i,l,m)] * self.t[(f[i - 1], e[j])]
                        for j in range(0,l) ]
                den = float(sum(num))

                likelihood += math.log(den)

                for j in range(0,l):

                    delta = num[j] / den

                    c1[(f[i - 1], e[j])] += delta
                    c2[(e[j],)]          += delta
                    c3[(j,i,l,m)]        += delta
                    c4[(i,l,m)]          += delta

        self.t = defaultdict(float,{k: v / c2[k[1:]] for k,v in c1.iteritems() if v > 0.0})
        self.q = defaultdict(float,{k: v / c4[k[1:]] for k,v in c3.iteritems() if v > 0.0})

        print("\rPass %2d: 100.00%% (Elapsed: %.2fs) (Likelihood: %.5f)" % (passnum,(time.time() - start),likelihood))


    def predict_alignment(self,e,f):
        l = len(e) + 1
        m = len(f) + 1
        e = [None] + e

        # for each french word:
        #  - compute a list of indices j of words in the english sentence,
        #    together with the probability of e[j] being aligned with f[i-1]
        #  - take the index j for the word with the _highest_ probability;

        def maximum_alignment(i):
            possible_alignments = [(j, self.t[(f[i - 1], e[j])] * self.q[(j, i, l, m)]) for j in range(0, l)]
            return max(possible_alignments, key=lambda x: x[1])[0]

        return [
            maximum_alignment(i)
            for i in range(1, m)]

    @classmethod
    def random(cls, corpus):
        return cls.with_generator(
            corpus, lambda n: np.random.dirichlet(np.ones(n), size=1)[0])


    @classmethod
    def uniform(cls,corpus):
        return cls.with_generator(corpus, lambda n: [1 / float(n)] * n)


    @classmethod
    def with_generator(cls,corpus,g):

        start = time.time()

        # "Compute all possible alignments..."
        lens   = set()
        aligns = defaultdict(set)

        for k, (f, e) in enumerate(corpus):

            if k % 1000 == 0:
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

            if k % 1000 == 0:
                stdout.write("\rInit    %6.2f%%" % (33 + ((33*k) / float(len(aligns)))))
                stdout.flush()
            k += 1

            p_values = g(len(aligns_to_e))
            t.update(zip(aligns_to_e,p_values))

        # "Compute initial probabilities for each distortion..."
        q = dict()
        for k, (l, m) in enumerate(lens):

            if k % 1000 == 0:
                stdout.write("\rInit    %6.2f%%" % (66 + ((33*k) / float(len(lens)))))
                stdout.flush()

            for i in range(1,m):
                p_values = g(l)
                for j in range(0,l):
                    q[(j,i,l,m)] = p_values[j]

        print "\rInit     100.00%% (Elapsed: %.2fs)" % (time.time() - start)

        return cls(t,q)
