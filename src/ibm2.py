from collections import defaultdict
from msgpack     import pack,unpack
from random      import random
from sys         import stdout

class IBM2:

    @staticmethod
    def load(stream):
        (t,q) = unpack(stream,use_list=False)
        return IBM2(defaultdict(float,t), defaultdict(float,q))

    def dump(self,stream):
        pack((self.t,self.q), stream)

    @staticmethod
    def random():
        return IBM2(defaultdict(random), defaultdict(random))

    def __init__(self, t, q):
        self.t = t
        self.q = q

    def em_train(self,corpus,n=10):
        for k in range(1,n+1):
            self.em_iter(corpus,passnum=k)
            stdout.write("\rPass %2d: 100.000%%\n" % k)
            stdout.flush()

    def em_iter(self,corpus,passnum=0):

        c1 = defaultdict(float) # ei aligned with fj
        c2 = defaultdict(float) # ei aligned with anything
        c3 = defaultdict(float) # wj aligned with wi
        c4 = defaultdict(float) # wi aligned with anything

        for k, (e, f) in enumerate(corpus):

            stdout.write("\rPass %2d: %7.3f%%" % (passnum, (100*k) / float(len(corpus))))
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


def read_corpus(path):
    with open(path,'r') as f:
        return [ ln.strip().split() for ln in f.readlines() ]


fr_corpus_path = '../data/training/hansards.36.2.f'
en_corpus_path = '../data/training/hansards.36.2.e'
model_path_1   = '../data/training/hansards.36.2.pass1.pack'
model_path_5   = '../data/training/hansards.36.2.pass5.pack'
model_path_10  = '../data/training/hansards.36.2.pass10.pack'
corpus         = zip(read_corpus(fr_corpus_path),read_corpus(en_corpus_path))


with open(model_path_1,'r') as stream:
    ibm = IBM2.load(stream)

ibm = IBM2.random()
ibm.em_train(corpus,n=4)

with open(model_path_5,'w') as stream:
    ibm.dump(stream)
