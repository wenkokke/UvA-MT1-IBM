import operator
import pickle
import random
import sys

#     def p_alignment(self,a,e,m):
#         """
#         Compute the probability of an alignment, conditional on an English word
#         and the length of the French sentence.
#
#         a -- alignment
#         e -- English sentence
#         m -- French sentence length
#         """
#         return 1 / float(len(e) + 1) ** m


class pRandom(dict):
    def __missing__(self, key):
        """
        Compute a random value, store it under the given key, and return it.
        """
        value = random.random()
        self[key] = value
        return value


class pByCount(dict):
    def __init__(self,count1,count2):
        self.count1 = count1
        self.count2 = count2

    def __repr__(self):
        return str({k: v for k,v in self.count1.iteritems() if v > 0.05})

    def __missing__(self,key):
        value = self.count1[key] / float(self.count2[key[1:]])
        self[key] = value
        return value


def at(sent,i):
    """
    Look up the word at the i'th position in a sentence.

    sent -- the given sentence (a list of words)
    i    -- the index, where 1 is the first word, 2 is the second, etc.
    """
    return sent[i - 1] if i >= 1 else None



class IBM2:
    """An implementation of the IBM2 model."""

    def __init__(self):
        self.t = None
        self.q = None

    def __repr__(self):
        return repr(self.t)

    def em_train(self,corpus,n=10):
        self.em_iteration_init()
        for i in range(n):
            self.em_iteration(corpus)

    def em_iteration_init(self):
        self.t = pRandom()
        self.q = pRandom()

    def em_iteration(self,corpus):
        """
        corpus -- list of tuples, consisting of a French sentence, an English
                  sentence
        """
        c1 = {} # ei aligned with fj
        c2 = {} # ei aligned with anything
        c3 = {} # wj aligned with wi
        c4 = {} # wi aligned with anything

        for k, (e, f) in enumerate(corpus):
            sys.stdout.write("\r%.3f%%" % ((100 * k) / float(len(corpus))))
            sys.stdout.flush()

            l = len(e)
            m = len(f)

            for i in range(1,m):

                numerators  = [self.q[(j,i,l,m)] * self.t[(at(f,i), at(e,j))]
                               for j in range(0,l)]
                denominator = float(sum(numerators))

                if denominator > 0:
                    for j in range(0,l):

                        delta = numerators[j] / denominator

                        # update counts for _words_:
                        k1     = (at(f,i), at(e,j))
                        c1[k1] = c1.setdefault(k1,0) + delta
                        k2     = k1[1:]
                        c2[k2] = c2.setdefault(k2,0) + delta

                        # update counts for _distortions_:
                        k3     = (j,i,l,m)
                        c3[k3] = c3.setdefault(k3,0) + delta
                        k4     = k3[1:]
                        c4[k4] = c4.setdefault(k4,0) + delta

        self.t = pByCount(c1,c2)
        self.q = pByCount(c3,c4)


    def naive(self,corpus):
        """
        corpus -- list of triples, consisting of a French sentence, an English
                  sentence and an alignment
        """
        c1 = {} # ei aligned with fj
        c2 = {} # ei aligned with anything
        c3 = {} # wj aligned with wi
        c4 = {} # wi aligned with anything

        for (e, f, a) in corpus:
            l = len(e)
            m = len(f)

            for i in range(1,m):
                for j in range(0,l):

                    # if delta is 1:
                    if at(a,i) == j:

                        # update counts for _words_:
                        if j >= 1:
                            k1     = (at(f,i),at(e,j))
                            c1[k1] = c1.setdefault(k1,0) + 1
                            k2     = (at(e,j),)
                            c2[k2] = c2.setdefault(k2,0) + 1

                        # update counts for _distortions_:
                        k3     = (j,i,l,m)
                        c3[k3] = c3.setdefault(k3,0) + 1
                        k4     = (i,l,m)
                        c4[k4] = c4.setdefault(k4,0) + 1

        self.t = pByCount(c1,c2)
        self.q = pByCount(c3,c4)

    def p_sentence(self,f,a,e,m):
        """
        Compute the probability of a French sentence, conditional on an
        alignment, an English sentence and the length of the French sentence.

        f -- French sentence
        a -- alignment
        e -- English sentence
        m -- French sentence length
        """
        return reduce(operator.mul,
                      [self.t(fi, at(e,a[i])) for i, fi in enumerate(f)])

    def p_alignment(self,a,e,m):
        """
        Compute the probability of an alignment, conditional on an English word
        and the length of the French sentence.

        a -- alignment
        e -- English sentence
        m -- French sentence length
        """
        l = len(e)
        return reduce(operator.mul,
                      [self.q(at(a,j),j,l,m) for j in range(1,l)])

    def p_sentence_with_alignment(self,f,a,e,m):
        """
        Compute the probability of a French sentence, together with an
        alignment, conditional on an English sentence and the length of
        the French sentence.

        f -- French sentence
        a -- alignment
        e -- English sentence
        m -- French sentence length
        """
        return self.p_alignment(a,e,m) * self.p_sentence(f,a,e,m)



def read_corpus(path):
    with open(path,'r') as f:
        return [ ln.strip().split() for ln in f.readlines() ]

#fr_corpus_path = '../data/training/hansards.36.2.f'
#en_corpus_path = '../data/training/hansards.36.2.e'
#pickle_path    = '../data/training/hansards.36.2.pkl'
#corpus         = zip(read_corpus(fr_corpus_path),read_corpus(en_corpus_path))
#
#ibm = IBM2()
#ibm.em_train(corpus,n=1)
#
#with open(pickle_path, 'w') as f:
#    pickle.dump(ibm, f)
