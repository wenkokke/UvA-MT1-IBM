# IBM Models 1 and 2

Python implementation of IBM Models 1 and 2.
Tested using the a French-English parallel
corpus from the HLT-NAACL 2003 Workshop.

We use various initializations for both model 1 and 2:

* Random initialization
* Uniform initialization
* Initialization of IBM model 2 with `t` from IBM model 1

We also added improvements to IBM model 1:

* Smoothing
* N-Null words

## Training

Train the model by executing.

```bash
python src/main.py
```

**Be aware** we will save intermediate states of the model
which may result in large data files (400MB+).

## Evaluation

The python code will dump `.eval` files which can be used with
the provided perl script: `data/test/eval/wa_eval_align.pl`.
Or use `bash eval.sh` to execute all evaluations at once.