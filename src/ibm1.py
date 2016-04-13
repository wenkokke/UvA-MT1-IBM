from collections import defaultdict,namedtuple
from itertools   import chain,product,repeat
from msgpack     import pack,unpack
from random      import random
from sys         import stdout
from os          import path

import operator
import numpy as np

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
            print("\rPass %2d: 100.00%%" % k)

    def em_iter(self,corpus,passnum=1):

        c1 = defaultdict(float) # ei aligned with fj
        c2 = defaultdict(float) # ei aligned with anything

        for k, (f, e) in enumerate(corpus):

            if k % 100 == 0:
                stdout.write("\rPass %2d: %6.2f%%" % (passnum, (100*k) / float(len(corpus))))
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
            k: (v + self.param.n) / (c2[k[1:]] + (n * self.param.v))
            for k,v in c1.iteritems() if v > 0.0 })


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

        # "Compute all possible alignments..."
        lens   = set()
        aligns = defaultdict(set)

        for k, (f, e) in enumerate(corpus):
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
            stdout.write("\rInit    %6.2f%%" % (50 + ((50*k) / float(len(aligns)))))
            stdout.flush()
            k += 1

            p_values = g(len(aligns_to_e))
            t.update(zip(aligns_to_e,p_values))

        print "\rInit     100.00%"

        return cls(t,param)



def read_corpus(path):
    """Read a file as a list of lists of words."""
    with open(path,'r') as f:
        return [ ln.strip().split() for ln in f ]


def main(corpus, mk_ibm, pack_path, corpus_name, n):

    ibm = None

    for s in range(0, n):
        curr_pack_path = path.join(pack_path , corpus_name + '.' + str(s  ) + '.pack')
        next_pack_path = path.join(pack_path , corpus_name + '.' + str(s+1) + '.pack')

        if not path.isfile(next_pack_path):

            if path.isfile(curr_pack_path):
                with open(curr_pack_path, 'r') as stream:
                    ibm = IBM.load(stream)
                    print "Loaded %s" % (curr_pack_path)

            else:
                if ibm is None:
                    ibm = mk_ibm()
                ibm.em_train(corpus, n=1, s=s)

                with open(curr_pack_path, 'w') as stream:
                    ibm.dump(stream)
                    print "Dumped %s" % (curr_pack_path)

            print_test_example(ibm)


def print_test_example(ibm):
    e = 'the government is doing what the Canadians want .'.split()
    f = 'le gouvernement fait ce que veulent les Canadiens .'.split()
    a = ibm.predict_alignment(e,f)

    print ' '.join(e)
    print ' '.join(f)
    e =  IBM.nones(ibm.param.q0,arg='NULL') + e
    print ' '.join([e[j] for j in a])


if __name__ == "__main__":

    data_path   = 'data'
    corpus_name = '10000'
    corpus_path = path.join(path.dirname(__file__), '..',
                                 data_path, 'training', corpus_name)
    fr_corpus_path   = corpus_path + '.f'
    en_corpus_path   = corpus_path + '.e'
    corpus = zip(read_corpus(fr_corpus_path), read_corpus(en_corpus_path))

    param = Param(q0 = 1000)
    main(corpus, lambda: IBM.random(corpus,param) , path.join(data_path,'model','ibm1','rand+100'), corpus_name, 20)
    param = Param(n = 0.01)
    main(corpus, lambda: IBM.random(corpus,param) , path.join(data_path,'model','ibm1','rand+n=0.01'), corpus_name, 20)
