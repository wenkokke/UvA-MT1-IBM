from collections import defaultdict,namedtuple
from itertools   import product,repeat
from msgpack     import pack,unpack
from sys         import stdout
import numpy as np
import time
import math


Param = namedtuple('Param',['q0','n','v'])
Param.__new__.__defaults__ = (1 , 0.01 , 100000)

class Param:
    def __init__(self,q0 = 1,n = 0, v = 100000):
        self.q0 = q0 # added number of NULL words
        self.n  = n  # smoothing ratio
        self.v  = v  # number of lexical items


class IBM:

    @classmethod
    def load(cls,stream):
        t = unpack(stream,use_list=False)
        return cls(defaultdict(float,t))

    def dump(self,stream):
        pack(self.t, stream)

    def __init__(self,t,param=None):
        self.t = t
        if param is None:
            self.param = Param(q0 = 1, n = 0.01, v = 100.000)
        else:
            self.param = param

    @staticmethod
    def nones(q0,arg=None):
        return list(repeat(arg,q0))

    def em_train(self,corpus,n=10,s=1):
        for k in range(s, n + s):
            self.em_iter(corpus, passnum=k)

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
                    [(self.t[(f[i - 1], e[j])]) / (m * l)
                     for j in range(0, l)])
            likelihood += math.log(score)

        print("\rLog-likelihood: %.5f (Elapsed: %.2fs)" % (likelihood, (time.time() - start)))

        return likelihood

    def em_iter(self,corpus,passnum=1):

        start = time.time()

        c1 = defaultdict(float) # ei aligned with fj
        c2 = defaultdict(float) # ei aligned with anything

        for k, (f, e) in enumerate(corpus):

            if k % 1000 == 0:
                stdout.write("\rPass %2d: %6.2f%%" % (passnum, (100 * k) / float(len(corpus))))
                stdout.flush()

            e = IBM.nones(self.param.q0) + e
            l = len(e)
            m = len(f) + 1
            q = 1 / float(len(e))

            for i in range(1,m):

                num = [ q * self.t[(f[i - 1], e[j])] for j in range(0,l) ]
                den = float(sum(num))

                for j in range(0,l):

                    delta = num[j] / den

                    c1[(f[i - 1], e[j])] += delta
                    c2[(e[j],)]          += delta

        self.t = defaultdict(float,{
            k: (v + self.param.n) / (c2[k[1:]] + (self.param.n * self.param.v))
            for k,v in c1.iteritems() if v > 0.0 })

        print("\rPass %2d: 100.00%% (Elapsed: %.2fs)" % (passnum, (time.time() - start)))


    def predict_alignment(self,e,f):
        e = IBM.nones(self.param.q0) + e
        l = len(e)
        m = len(f) + 1

        # for each french word:
        #  - compute a list of indices j of words in the english sentence,
        #    together with the probability of e[j] being aligned with f[i-1]
        #  - take the index j for the word with the _highest_ probability;
        return [
            max([ (j, self.t[(f[i - 1], e[j])]) for j in range(0,l) ]
                , key = lambda x: x[1])[0]
            for i in range(1,m) ]

    @classmethod
    def random(cls,corpus,param):
        return cls.with_generator(
            corpus, lambda n: np.random.dirichlet(np.ones(n),size=1)[0],param)

    @classmethod
    def uniform(cls,corpus,param):
        return cls.with_generator(corpus, lambda n: [1 / float(n)] * n,param)

    @classmethod
    def with_generator(cls,corpus,g,param):

        start = time.time()

        # "Compute all possible alignments..."
        lens   = set()
        aligns = defaultdict(set)

        for k, (f, e) in enumerate(corpus):

            if k % 1000 == 0:
                stdout.write("\rInit    %6.2f%%" % ((50*k) / float(len(corpus))))
                stdout.flush()

            e = IBM.nones(param.q0) + e
            lens.add((len(e), len(f) + 1))

            for (f, e) in product(f, e):
                aligns[e].add((f, e))

        # "Compute initial probabilities for each alignment..."
        k = 0
        t = dict()
        for e, aligns_to_e in aligns.iteritems():

            if k % 1000 == 0:
                stdout.write("\rInit    %6.2f%%" % (50 + ((50*k) / float(len(aligns)))))
                stdout.flush()
            k += 1

            p_values = g(len(aligns_to_e))
            t.update(zip(aligns_to_e,p_values))

        print "\rInit     100.00%% (Elapsed: %.2fs)" % (time.time() - start)

        return cls(t,param)