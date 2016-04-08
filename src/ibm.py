import operator


class IBM1:
    """An implementation of the IBM1 model."""

    def __init__(self):
        self.count1 = {}
        self.count2 = {}
        self.count3 = {}
        self.count4 = {}

    def __repr__(self):
        return "\n".join([ str(self.count1),
                           str(self.count2),
                           str(self.count3),
                           str(self.count4)])

    def p_alignment(self,a,e,m):
        """
        Compute the probability of an alignment, conditional on an English word
        and the length of the French sentence.

        a -- alignment
        e -- English sentence
        m -- French sentence length
        """
        return 1 / float(len(e) + 1) ** m

    def p_word(self,f,e):
        """
        Compute the probability of a French word, conditional on an English word.

        f -- French word
        e -- English word
        """
        self.count[(f,e)] / float(self.count[e])

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
                      [self.p_word(fi, e[a[i]]) for i, fi in enumerate(f)])

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


class IBM2(IBM1):
    """An implementation of the IBM2 model."""

    def ml(self,corpus):
        """
        corpus -- list of triples, consisting of a French sentence, an English
                  sentence and an alignment
        """

        for (f, e, a) in corpus:
            f = f.split()
            e = e.split()
            l = len(e)
            m = len(f)

            for i in range(1,m):
                for j in range(0,l):

                    # if delta is 1:
                    if a[i - 1] == j:

                        # update counts for _words_:
                        if j >= 1:
                            key1 = (e[j - 1], f[i - 1]) # ei aligned with fj
                            key2 = (e[j - 1])           # ei aligned with anything
                            self.count1[key1] = self.count1.setdefault(key1,0) + 1
                            self.count2[key2] = self.count2.setdefault(key2,0) + 1

                        # update counts for _distortions_:
                        key3 = (j,i,l,m) # wj aligned with wi
                        key4 = (i,l,m)   # wi aligned with anything
                        self.count3[key3] = self.count3.setdefault(key3,0) + 1
                        self.count4[key4] = self.count4.setdefault(key4,0) + 1

    def p_distortion(self,i,j,l,m):
        """
        Compute the probability that the j'th French word is connected to the
        i'th English word.

        i -- position of the English word
        j -- position of the French word
        l -- length of the English sentence
        m -- length of the French sentence
        """
        return self.count[(j,i,l,m)] / float(self.count[(i,l,m)])


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
                      [self.p_distortion(a[j - 1],j,l,m) for j in range(1,l)])




ibm    = IBM2()
corpus = [ ("the wild cat"                   , "le chat sauvage"                          , [1,3,2])
         , ("the whale is a house"           , "la balaine est un maison"                 , [1,2,3,4,5])
         , ("where is the cat"               , "ou est le chat"                           , [1,2,3,4])
         , ("the large whale"                , "la balaine grande"                        , [1,3,2])
         , ("cats and whales love the house" , "des chats et des balaines aime le maison" , [2,3,5,6,7,8])
         ]
ibm.ml(corpus)
print ibm
