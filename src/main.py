# coding: utf-8
import itertools
from msgpack     import pack,unpack
from os          import path

import os
import matplotlib.pyplot as plt
import ibm2
import ibm1


class Results:

    @classmethod
    def load(cls, stream):
        (log_likelihoods, recalls, precisions, aers) = unpack(stream, use_list=True)
        return cls(log_likelihoods, recalls, precisions, aers)

    def dump(self, stream):
        pack((self.log_likelihoods, self.recalls, self.precisions, self.aers), stream)

    def __init__(self, log_likelihoods, recalls, precisions, aers):
        self.log_likelihoods = log_likelihoods
        self.recalls = recalls
        self.precisions = precisions
        self.aers = aers

    def save_plot_log_likelihood(self, packs_path, corpus_name):
        plt.plot(self.log_likelihoods)
        plt.savefig(path.join(packs_path, corpus_name + '.log_likelihood.png'))
        plt.clf()

    def log_likelihood(self, s, log_likelihood):
        if len(self.log_likelihoods) > s:
            self.log_likelihoods[s] = log_likelihood
        else:
            self.log_likelihoods.append(log_likelihood)


def read_corpus(path):
    """Read a file as a list of lists of words."""
    with open(path,'r') as f:
        return [ ln.strip().split() for ln in f ]


def run(corpus, ibm_cls, ibm_init, packs_path, corpus_name, n):

    ibm = None

    if not path.isdir(packs_path):
        os.makedirs(packs_path)

    for s in range(0, n + 1):
        curr_pack_path = path.join(packs_path , corpus_name + '.' + str(s  ) + '.pack')
        next_pack_path = path.join(packs_path , corpus_name + '.' + str(s+1) + '.pack')

        if s == n and path.isfile(curr_pack_path):
            continue

        if not path.isfile(next_pack_path) or not path.isfile(curr_pack_path):

            if path.isfile(curr_pack_path):
                with open(curr_pack_path, 'r') as stream:
                    ibm = ibm_cls.load(stream)
                    print "Loaded %s" % (curr_pack_path)

            else:
                if ibm is None:
                    ibm = ibm_init()
                else:
                    ibm.em_train(corpus, n=1, s=s)

                with open(curr_pack_path, 'w') as stream:
                    ibm.dump(stream)
                    print "Dumped %s" % (curr_pack_path)

    if ibm is not None:
        build_results(corpus, ibm_cls, packs_path, corpus_name, n)


def print_test_example(ibm):
    e = 'the government is doing what the Canadians want .'.split()
    f = 'le gouvernement fait ce que veulent les Canadiens .'.split()

    a = ibm.predict_alignment(e,f)

    print ' '.join(e)
    print ' '.join(f)
    e = ['NULL'] + e
    print ' '.join([e[j] for j in a])


def build_results(corpus, ibm_cls, packs_path, corpus_name, n):

    if not path.isdir(packs_path):
        os.makedirs(packs_path)

    results_path = path.join(packs_path, corpus_name + '.results.pack')

    if path.isfile(results_path):
        with open(results_path, 'r') as stream:
            results = Results.load(stream)
    else:
        results = Results([], [], [], [])

    for s in range(0, n + 1):
        pack_path = path.join(packs_path, corpus_name + '.' + str(s) + '.pack')

        if not path.isfile(pack_path):
            print "%s does not exist" % (pack_path)
            continue

        with open(pack_path, 'r') as stream:
            ibm = ibm_cls.load(stream)
            print "Loaded %s" % (pack_path)
            results.log_likelihood(s, ibm.log_likelihood(corpus))

        with open(results_path, 'w') as stream:
            results.dump(stream)
            print "Dumped %s" % (results_path)

    results.save_plot_log_likelihood(packs_path, corpus_name)

if __name__ == "__main__":

    data_path   = path.join(path.dirname(__file__), '..', 'data')
    corpus_name = 'hansards.36.2' # hansards.36.2
    corpus_path = path.join(data_path, 'training', corpus_name)
    fr_corpus_path   = corpus_path + '.f'
    en_corpus_path   = corpus_path + '.e'
    en_corpus = read_corpus(en_corpus_path)
    en_vocabulary_len = len(set(itertools.chain(*en_corpus)))
    corpus = zip(read_corpus(fr_corpus_path), en_corpus)

    # ibm = ibm2.IBM
    #
    # run(corpus, ibm, lambda: ibm.uniform(corpus), path.join(data_path,'model','ibm2','uniform'), corpus_name, 20)
    # run(corpus, ibm, lambda: ibm.random(corpus), path.join(data_path,'model','ibm2','random1'), corpus_name, 20)
    # run(corpus, ibm, lambda: ibm.random(corpus), path.join(data_path,'model','ibm2','random2'), corpus_name, 20)
    # run(corpus, ibm, lambda: ibm.random(corpus), path.join(data_path,'model','ibm2','random3'), corpus_name, 20)

    ibm = ibm1.IBM

    param = ibm1.Param()
    run(corpus, ibm, lambda: ibm.uniform(corpus, param), path.join(data_path, 'model', 'ibm1', 'uniform'), corpus_name, 20)
    param = ibm1.Param()
    run(corpus, ibm, lambda: ibm.random(corpus, param), path.join(data_path, 'model', 'ibm1', 'random1'), corpus_name, 20)
    param = ibm1.Param()
    run(corpus, ibm, lambda: ibm.random(corpus, param), path.join(data_path, 'model', 'ibm1', 'random2'), corpus_name, 20)
    param = ibm1.Param()
    run(corpus, ibm, lambda: ibm.random(corpus, param), path.join(data_path, 'model', 'ibm1', 'random3'), corpus_name, 20)

    param = ibm1.Param(n=0.01,v=en_vocabulary_len)
    run(corpus, ibm, lambda: ibm.random(corpus, param), path.join(data_path, 'model', 'ibm1', 'random-n0.01'), corpus_name, 20)
    param = ibm1.Param(n=0.005, v=en_vocabulary_len)
    run(corpus, ibm, lambda: ibm.random(corpus, param), path.join(data_path, 'model', 'ibm1', 'random-n0.005'), corpus_name, 20)
    param = ibm1.Param(n=0.0005, v=en_vocabulary_len)
    run(corpus, ibm, lambda: ibm.random(corpus, param), path.join(data_path, 'model', 'ibm1', 'random-n0.0005'), corpus_name, 20)

    param = ibm1.Param(q0=2)
    run(corpus, ibm, lambda: ibm.random(corpus, param), path.join(data_path, 'model', 'ibm1', 'random-q02'), corpus_name, 20)
    param = ibm1.Param(q0=3)
    run(corpus, ibm, lambda: ibm.random(corpus, param), path.join(data_path, 'model', 'ibm1', 'random-q03'), corpus_name, 20)
