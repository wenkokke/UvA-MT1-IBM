# coding: utf-8
import itertools
from msgpack     import pack,unpack
from os          import path

import os
import matplotlib.pyplot as plt
import ibm2
import ibm1


def read_corpus(path):
    """Read a file as a list of lists of words."""
    with open(path,'r') as f:
        return [ ln.strip().split() for ln in f ]


def run(corpus, ibm_cls, ibm_init, packs_path, corpus_name, n, skip_if_completed=True):

    ibm = None

    if not path.isdir(packs_path):
        os.makedirs(packs_path)

    for s in range(0, n + 1):
        curr_pack_path = path.join(packs_path , corpus_name + '.' + str(s  ) + '.pack')
        next_pack_path = path.join(packs_path , corpus_name + '.' + str(s+1) + '.pack')

        if skip_if_completed is True and s == n and path.isfile(curr_pack_path):
            continue

        if not skip_if_completed or not path.isfile(next_pack_path) or not path.isfile(curr_pack_path):

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

        test_model(ibm, packs_path, corpus_name, s)

    return ibm


def print_test_example(ibm):
    e = 'the government is doing what the Canadians want .'.split()
    f = 'le gouvernement fait ce que veulent les Canadiens .'.split()

    a = ibm.predict_alignment(e,f)

    print ' '.join(e)
    print ' '.join(f)
    e = ['NULL'] + e
    print ' '.join([e[j] for j in a])


def test_model(ibm, eval_data_path, corpus_name, s):

    test_path = path.join(path.dirname(__file__), '..', 'data', 'test')
    test_corpus_name = 'test'
    test_corpus_path = path.join(test_path, 'test', test_corpus_name)
    fr_test_corpus_path = test_corpus_path + '.f'
    en_test_corpus_path = test_corpus_path + '.e'
    test_corpus = zip(read_corpus(fr_test_corpus_path), read_corpus(en_test_corpus_path))

    if not path.isdir(eval_data_path):
        os.makedirs(eval_data_path)

    handle = open(path.join(eval_data_path, corpus_name + '.' + str(s) + '.eval'), 'w')

    for i,(f,e) in enumerate(test_corpus):

        for j,a in enumerate(ibm.predict_alignment(f,e)):
            if a is not 0:
                line = "%04d %d %d" % (i+1, a, j+1)
                handle.write(line + "\n")

    handle.close()

if __name__ == "__main__":

    data_path   = path.join(path.dirname(__file__), '..', 'data')
    corpus_name = 'hansards.36.2' # hansards.36.2
    corpus_path = path.join(data_path, 'training', corpus_name)
    fr_corpus_path   = corpus_path + '.f'
    en_corpus_path   = corpus_path + '.e'
    en_corpus = read_corpus(en_corpus_path)
    en_vocabulary_len = len(set(itertools.chain(*en_corpus)))
    corpus = zip(read_corpus(fr_corpus_path), en_corpus)

    ibm = ibm1.IBM
    param = ibm1.Param()
    model = run(corpus, ibm, lambda: ibm.random(corpus,param), path.join(data_path, 'model', 'ibm1', 'uniform'), corpus_name, 5, False)

    def ibm2_with_ibm1(ibm,model1,corpus):
        model2 = ibm.uniform(corpus)
        model2.t = model1.t
        return model2


    ibm = ibm2.IBM
    model_path = path.join(data_path, 'model', 'ibm2', 'ibm1-5')
    model = run(corpus, ibm, lambda: ibm2_with_ibm1(ibm,model,corpus), model_path, corpus_name, 20, False)

    ibm = ibm2.IBM

    run(corpus, ibm, lambda: ibm.uniform(corpus), path.join(data_path,'model','ibm2','uniform'), corpus_name, 20, False)
    run(corpus, ibm, lambda: ibm.random(corpus), path.join(data_path,'model','ibm2','random1'), corpus_name, 20, False)
    run(corpus, ibm, lambda: ibm.random(corpus), path.join(data_path,'model','ibm2','random2'), corpus_name, 20, False)
    run(corpus, ibm, lambda: ibm.random(corpus), path.join(data_path,'model','ibm2','random3'), corpus_name, 20, False)

    # ibm = ibm1.IBM
    #
    # param = ibm1.Param()
    # run(corpus, ibm, lambda: ibm.uniform(corpus, param), path.join(data_path, 'model', 'ibm1', 'uniform'), corpus_name, 20, False)
    # param = ibm1.Param()
    # run(corpus, ibm, lambda: ibm.random(corpus, param), path.join(data_path, 'model', 'ibm1', 'random1'), corpus_name, 20, False)
    # param = ibm1.Param()
    # run(corpus, ibm, lambda: ibm.random(corpus, param), path.join(data_path, 'model', 'ibm1', 'random2'), corpus_name, 20, False)
    # param = ibm1.Param()
    # run(corpus, ibm, lambda: ibm.random(corpus, param), path.join(data_path, 'model', 'ibm1', 'random3'), corpus_name, 20, False)
    #
    # param = ibm1.Param(n=0.01,v=en_vocabulary_len)
    # run(corpus, ibm, lambda: ibm.random(corpus, param), path.join(data_path, 'model', 'ibm1', 'random-n0.01'), corpus_name, 20, False)
    # param = ibm1.Param(n=0.005, v=en_vocabulary_len)
    # run(corpus, ibm, lambda: ibm.random(corpus, param), path.join(data_path, 'model', 'ibm1', 'random-n0.005'), corpus_name, 20, False)
    # param = ibm1.Param(n=0.0005, v=en_vocabulary_len)
    # run(corpus, ibm, lambda: ibm.random(corpus, param), path.join(data_path, 'model', 'ibm1', 'random-n0.0005'), corpus_name, 20, False)
    #
    # param = ibm1.Param(q0=2)
    # run(corpus, ibm, lambda: ibm.random(corpus, param), path.join(data_path, 'model', 'ibm1', 'random-q02'), corpus_name, 20, False)
    # param = ibm1.Param(q0=3)
    # run(corpus, ibm, lambda: ibm.random(corpus, param), path.join(data_path, 'model', 'ibm1', 'random-q03'), corpus_name, 20, False)
