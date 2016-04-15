# coding: utf-8
from os import path

import itertools
import os
import ibm2
import ibm1


def read_corpus(path):
    """Read a file as a list of lists of words."""

    with open(path,'r') as f:
        return [ ln.strip().split() for ln in f ]


def run(corpus, ibm_cls, ibm_init, packs_path, corpus_name, n):
    """Run n iterations of the EM algorithm on a certain corpus and save all intermediate and final results"""

    model = None

    if not path.isdir(packs_path):
        os.makedirs(packs_path)

    # Iterations
    for s in range(0, n + 1):
        curr_pack_path = path.join(packs_path , corpus_name + '.' + str(s  ) + '.pack')
        next_pack_path = path.join(packs_path , corpus_name + '.' + str(s+1) + '.pack')

        # Execute iteration if not already dumped to pack file
        if not path.isfile(next_pack_path) or not path.isfile(curr_pack_path):

            if path.isfile(curr_pack_path):
                with open(curr_pack_path, 'r') as stream:
                    model = ibm_cls.load(stream)
                    print "Loaded %s" % curr_pack_path

            else:
                if model is None:
                    model = ibm_init()
                else:
                    (likelihood, time) = model.em_iter(corpus, s)

                    # Save likelihood and time results so separate file
                    with open(path.join(packs_path, corpus_name + '.results'), "a") as results_handle:
                        results_handle.write("%d,%.4f,%.5f\n" % (s, time, likelihood))

                with open(curr_pack_path, 'w') as stream:
                    model.dump(stream)
                    print "Dumped %s" % curr_pack_path

        # Generate evaluation file for testing the model
        if model is not None:
            test_model(model, packs_path, corpus_name, s)

    return model


def test_model(model, eval_data_path, corpus_name, s):
    """
    Test a model against the provided test set by generating an evaluation file
    This file can be used by the provided 'wa_eval_align.pl'
    """

    test_path = path.join(path.dirname(__file__), '..', 'data', 'test')
    test_corpus_name = 'test'
    test_corpus_path = path.join(test_path, 'test', test_corpus_name)
    fr_test_corpus_path = test_corpus_path + '.f'
    en_test_corpus_path = test_corpus_path + '.e'
    test_corpus = zip(read_corpus(fr_test_corpus_path), read_corpus(en_test_corpus_path))

    if not path.isdir(eval_data_path):
        os.makedirs(eval_data_path)

    handle = open(path.join(eval_data_path, corpus_name + '.' + str(s) + '.eval'), 'w')

    for i, (f, e) in enumerate(test_corpus):

        for j, a in enumerate(model.viterbi_alignment(f, e)):
            if a is not 0:
                line = "%04d %d %d" % (i+1, a, j+1)
                handle.write(line + "\n")

    handle.close()


def print_test_example(ibm):
    """Prints the alignment results of a toy example"""

    f = 'le gouvernement fait ce que veulent les Canadiens .'.split()
    e = 'the government is doing what the Canadians want .'.split()

    a = ibm.viterbi_alignment(f,e)

    print ' '.join(e)
    print ' '.join(f)
    e = ['NULL'] + e
    print ' '.join([e[j] for j in a])


def main():
    """Program entry point"""

    data_path = path.join(path.dirname(__file__), '..', 'data')
    corpus_name = 'hansards.36.2'  # hansards.36.2
    corpus_path = path.join(data_path, 'training', corpus_name)
    fr_corpus_path = corpus_path + '.f'
    en_corpus_path = corpus_path + '.e'
    en_corpus = read_corpus(en_corpus_path)
    en_vocabulary_len = len(set(itertools.chain(*en_corpus)))
    corpus = zip(read_corpus(fr_corpus_path), en_corpus)

    # Train IBM2 by setting t as the output of 5 iterations of IBM1
    ibm = ibm1.IBM
    model = run(corpus, ibm, lambda: ibm.random(corpus), path.join(data_path, 'model', 'ibm1', 'uniform'), corpus_name, 5)

    def ibm2_with_ibm1(ibm, model1, corpus):
        model2 = ibm.uniform(corpus)
        model2.t = model1.t
        return model2

    ibm = ibm2.IBM
    model = run(corpus, ibm, lambda: ibm2_with_ibm1(ibm, model, corpus),
                path.join(data_path, 'model', 'ibm2', 'ibm1-5'), corpus_name, 20)

    # Train IBM2 with random and uniform initialization
    run(corpus, ibm, lambda: ibm.uniform(corpus), path.join(data_path, 'model', 'ibm2', 'uniform'), corpus_name, 20)
    run(corpus, ibm, lambda: ibm.random(corpus), path.join(data_path, 'model', 'ibm2', 'random1'), corpus_name, 20)
    run(corpus, ibm, lambda: ibm.random(corpus), path.join(data_path, 'model', 'ibm2', 'random2'), corpus_name, 20)
    run(corpus, ibm, lambda: ibm.random(corpus), path.join(data_path, 'model', 'ibm2', 'random3'), corpus_name, 20)

    # Train IBM1 with random and uniform initialization
    ibm = ibm1.IBM
    run(corpus, ibm, lambda: ibm.uniform(corpus), path.join(data_path, 'model', 'ibm1', 'uniform'), corpus_name, 20)
    run(corpus, ibm, lambda: ibm.random(corpus), path.join(data_path, 'model', 'ibm1', 'random1'), corpus_name, 20)
    run(corpus, ibm, lambda: ibm.random(corpus), path.join(data_path, 'model', 'ibm1', 'random2'), corpus_name, 20)
    run(corpus, ibm, lambda: ibm.random(corpus), path.join(data_path, 'model', 'ibm1', 'random3'), corpus_name, 20)

    # Train IBM1 with smoothing
    param = ibm1.Param(n=0.01, v=en_vocabulary_len)
    run(corpus, ibm, lambda: ibm.random(corpus, param), path.join(data_path, 'model', 'ibm1', 'random-n0.01'), corpus_name, 20)
    param = ibm1.Param(n=0.005, v=en_vocabulary_len)
    run(corpus, ibm, lambda: ibm.random(corpus, param), path.join(data_path, 'model', 'ibm1', 'random-n0.005'), corpus_name, 20)
    param = ibm1.Param(n=0.0005, v=en_vocabulary_len)
    run(corpus, ibm, lambda: ibm.random(corpus, param), path.join(data_path, 'model', 'ibm1', 'random-n0.0005'), corpus_name, 20)

    # Train IBM1 with aditional null words
    param = ibm1.Param(q0=2)
    run(corpus, ibm, lambda: ibm.random(corpus, param), path.join(data_path, 'model', 'ibm1', 'random-q02'), corpus_name, 20)
    param = ibm1.Param(q0=3)
    run(corpus, ibm, lambda: ibm.random(corpus, param), path.join(data_path, 'model', 'ibm1', 'random-q03'), corpus_name, 20)


if __name__ == "__main__":
    main()

