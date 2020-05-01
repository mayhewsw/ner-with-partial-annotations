# NER with Partial Annotations
Code for the CoNLL2019 paper on [NER with Partial Annotations](https://cogcomp.seas.upenn.edu/papers/MCTR19.pdf). See also the paper in the [ACL Anthology](https://www.aclweb.org/anthology/K19-1060/).




# Installation

NOTE: this code uses AllenNLP 0.8.4 and ccg_nlpy. 
AllenNLP in particular has changed a lot since we 
wrote this code, so getting the right version is 
important!

```bash
$ pip install ccg_nlpy allennlp==0.8.4
```



# Data & Embeddings

You can see some sample data in [`data/eng`](data/eng). These files have 
TextAnnotation format, from [ccg_nlpy](https://github.com/CogComp/cogcomp-nlpy).

You will need to set paths in [`utils.py`](mylib/utils.py) for the 
embeddings, and the data.


If you want to use BERT instead regular embeddings, change `USING_BERT`
in `utils.py` to `true`.

# Running

For the main results:

```bash
$ python main_ours.py <lang>
```

For the others, the names should be self-explanatory!



# Citation

If you use this code, please cite us!

```
@inproceedings{MCTR19,
    author = {Stephen Mayhew and Snigdha Chaturvedi and Chen-Tse Tsai and Dan Roth},
    title = {{Named Entity Recognition with Partially Annotated Training Data}},
    booktitle = {Proc. of the Conference on Computational Natural Language Learning (CoNLL)},
    year = {2019},
    url = "https://cogcomp.seas.upenn.edu/papers/MCTR19.pdf",
    funding = {LORELEI},
}
```
