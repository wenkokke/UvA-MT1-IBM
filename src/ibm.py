import random
import operator

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


class tByCount(dict):
    """
    Compute the probability of a French word, conditional on an English word.

    f -- French word
    e -- English word
    """
    def __init__(self,count1,count2):
        self.count1 = count1
        self.count2 = count2

    def __repr__(self):
        return "\n".join([str(self.count1), str(self.count2)])

    def __missing__(self, key):
        """
        key -- a tuple (e,f) of an English and a French word
        """
        value = self.count1.get(key,0) / float(self.count2.get(key[1],0))
        self[key] = value
        return value


class qByCount(dict):
    """
    Compute the probability that the j'th French word is connected to the
    i'th English word. (Memoized if not yet computed.)

    i -- position of the English word
    j -- position of the French word
    l -- length of the English sentence
    m -- length of the French sentence
    """
    def __init__(self,count3,count4):
        self.count3 = count3
        self.count4 = count4

    def __repr__(self):
        return "\n".join([str(self.count3), str(self.count4)])

    def __missing__(self,key):
        """
        key -- a quadruple of (j,i,l,m)
        i   -- position of the English word
        j   -- position of the French word
        l   -- length of the English sentence
        m   -- length of the French sentence
        """
        value = self.count3.get(key,0) / float(self.count4.get(key[1:],0))
        self[key] = value
        return value


class IBM2:
    """An implementation of the IBM2 model."""

    def __init__(self):
        self.t = None
        self.q = None

    def __repr__(self):
        return "\n".join([repr(self.t), repr(self.q)])

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

        for (e, f) in corpus:
            l = len(e)
            m = len(f)

            for i in range(1,m):

                numerators  = [self.q[(j,i,l,m)] * self.t[(f[i - 1],e[j - 1])]
                               for j in range(0,l)]
                denominator = float(sum(numerators))

                if denominator > 0:
                    for j in range(0,l):

                        delta = numerators[j] / denominator

                        # update counts for _words_:
                        k1     = (e[j - 1], f[i - 1])
                        c1[k1] = c1.setdefault(k1,0) + delta
                        k2     = (e[j - 1])
                        c2[k2] = c2.setdefault(k2,0) + delta

                        # update counts for _distortions_:
                        k3     = (j,i,l,m)
                        c3[k3] = c3.setdefault(k3,0) + delta
                        k4     = (i,l,m)
                        c4[k4] = c4.setdefault(k4,0) + delta

        self.t = tByCount(c1,c2)
        self.q = qByCount(c3,c4)


    def em_naive(self,corpus):
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
                    if a[i - 1] == j:

                        # update counts for _words_:
                        if j >= 1:
                            k1     = (e[j - 1], f[i - 1])
                            c1[k1] = c1.setdefault(k1,0) + 1
                            k2     = (e[j - 1])
                            c2[k2] = c2.setdefault(k2,0) + 1

                        # update counts for _distortions_:
                        k3     = (j,i,l,m)
                        c3[k3] = c3.setdefault(k3,0) + 1
                        k4     = (i,l,m)
                        c4[k4] = c4.setdefault(k4,0) + 1

        self.t = tByCount(c1,c2)
        self.q = qByCount(c3,c4)

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
                      [self.t(fi, e[a[i]]) for i, fi in enumerate(f)])

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
                      [self.q(a[j - 1],j,l,m) for j in range(1,l)])

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



corpus = [ (f.split(), e.split(), a) for (f, e, a) in
           [   ("le chat sauvage"                          , "the wild cat"                   , [1,3,2])
             , ("la balaine est un maison"                 , "the whale is a house"           , [1,2,3,4,5])
             , ("ou est le chat"                           , "where is the cat"               , [1,2,3,4])
             , ("la balaine grande"                        , "the large whale"                , [1,3,2])
             , ("des chats et des balaines aime le maison" , "cats and whales love the house" , [2,3,5,6,7,8])
           ]]

ibm = IBM2()
ibm.em_train([item[:2] for item in corpus])
print ibm
