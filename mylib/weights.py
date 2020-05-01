from collections import defaultdict
from typing import List
import numpy as np
from allennlp.data import Instance

import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def add_weights(data: List[Instance], bstar):
    """
    This will add weights to data, based on frequency.
    :param data:
    :return:
    """

    freqdict = defaultdict(int)
    totaltoks = 0
    tags = 0

    for inst in data:
        for tag in inst["metadata"]["orig_tags"]:
            if tag != "O":
                tags += 1

        for tok in inst["tokens"]:
            freqdict[tok.text.lower()] += 1
            totaltoks += 1

    neg_freqs = 0

    # now get the frequency of all negative elements
    for inst in data:
        toks = inst["tokens"]
        for i, tag in enumerate(inst["metadata"]["orig_tags"]):
            freq = freqdict[toks[i].text.lower()]
            if tag == "O":
                neg_freqs += freq

    # Normalize in log space
    for w in freqdict:
        freqdict[w] = np.log(freqdict[w] / float(totaltoks))

    mx = max(freqdict.values())
    mn = min(freqdict.values())

    for w in freqdict:
        freqdict[w] = (freqdict[w] - mn) / (mx-mn)

    sf = sorted(freqdict.items(), key=lambda p: p[1], reverse=True)

    # now get the frequency of all negative elements
    for inst in data:
        toks = inst["tokens"]
        tags = inst["metadata"]["orig_tags"]
        for i, tag in enumerate(tags):

            # this is the window weighting.
            if (i > 0 and tags[i-1] != "O") or (i < (len(tags)-1) and tags[i+1] != "O"):
                continue

            if tag == "O":
                freq = freqdict[toks[i].text.lower()]
                num_tags = len(inst["tags"][i].array)

                marginals = np.zeros(num_tags)

                # first element is always O
                # this value ranges from 1/num_tags to 1. Clamp at 1.
                marginals[0] = min(freq + 1/num_tags, 1)
                remainder = (1 - marginals[0]) / (num_tags-1)

                for k in range(1, num_tags):
                    marginals[k] = remainder

                # gotta do this maximum thing to avoid -inf
                inst["tags"][i].array = np.maximum(np.log(marginals), -10000)

