from collections import defaultdict
from itertools   import chain,product
from msgpack     import pack,unpack
from random      import random
from sys         import stdout
from os          import path

import numpy as np

class IBM2:

    @classmethod
    def load(cls,stream):
        (t,q) = unpack(stream,use_list=False)
        return cls(defaultdict(float,t), defaultdict(float,q))

    def dump(self,stream):
        pack((self.t,self.q), stream)

    def __init__(self, t, q):
        self.t = t
        self.q = q

    def em_train(self,corpus,n=10):
        for k in range(1,n+1):
            self.em_iter(corpus,passnum=k)
            print("\rPass %2d: 100.00%%" % k)

    def em_iter(self,corpus,passnum=1):

        c1 = defaultdict(float) # ei aligned with fj
        c2 = defaultdict(float) # ei aligned with anything
        c3 = defaultdict(float) # wj aligned with wi
        c4 = defaultdict(float) # wi aligned with anything

        for k, (f, e) in enumerate(corpus):

            if k % 100 == 0:
                stdout.write("\rPass %2d: %6.2f%%" % (passnum, (100*k) / float(len(corpus))))
                stdout.flush()

            l = len(e)
            m = len(f)
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

    @classmethod
    def random(cls,corpus):
        return cls.with_generator(
            corpus, lambda n: np.random.dirichlet(np.ones(n),size=1)[0])

    @classmethod
    def uniform(cls,corpus):
        return cls.with_generator(corpus, lambda n: 1 / float(n))

    @classmethod
    def with_generator(cls,corpus,g):

        # "Compute all possible alignments..."
        lens   = set()
        aligns = defaultdict(set)

        for k, (f, e) in enumerate(corpus):
            stdout.write("\rInit    %6.2f%%" % ((33*k) / float(len(corpus))))
            stdout.flush()

            lens.add((len(e), len(f)))

            for (f, e) in product(f, [None] + e):
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


if __name__ == "__main__":
    src = path.dirname(__file__)
    mini_corpus_txt  = path.join(src,'../data/mini_corpus.txt')
    mini_corpus_pack = path.join(src,'../data/mini_corpus.pack')

    with open(mini_corpus_txt,'r') as f:
        corpus = [ln.strip().split() for ln in f.readlines()]
        corpus = zip(corpus[0:][::3],corpus[1:][::3])

        ibm = IBM2.random(corpus)
        ibm.em_train(corpus,n=20)

    with open(mini_corpus_pack,'w') as f:
        ibm.dump(f)

    for k,v in ibm.t.iteritems():
        if v > 0.05:
            print k, v

#if __name__ == "__main__":
#
#    fr_corpus_path = path.join(path.dirname(__file__),'../data/training/hansards.36.2.f')
#    en_corpus_path = path.join(path.dirname(__file__),'../data/training/hansards.36.2.e')
#    model_path_0   = path.join(path.dirname(__file__),'../data/hansards.36.2.rand.pack')
#    model_path_1   = path.join(path.dirname(__file__),'../data/hansards.36.2.rand.pass1.pack')
#    model_path_5   = path.join(path.dirname(__file__),'../data/hansards.36.2.rand.pass5.pack')
#    model_path_10  = path.join(path.dirname(__file__),'../data/hansards.36.2.rand.pass10.pack')
#    model_path_20  = path.join(path.dirname(__file__),'../data/hansards.36.2.rand.pass20.pack')
#    corpus         = zip(read_corpus(fr_corpus_path),read_corpus(en_corpus_path))
#    corpus         = corpus
#
#    ibm = IBM2.random(corpus)
#    with open(model_path_0,'w') as f:
#        ibm.dump(f)
#    ibm.em_train(corpus,n=1)
#    with open(model_path_1,'w') as f:
#        ibm.dump(f)
