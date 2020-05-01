from typing import Dict, List, Sequence, Iterable
import os

from overrides import overrides
import ccg_nlpy as ccg

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField, ArrayField, MultiLabelField, \
    ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
import logging
import numpy as np
import random

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ''
    if empty_line:
        return True
    else:
        first_token = line.split()[0]
        if first_token == "-DOCSTART-":  # pylint: disable=simplifiable-if-statement
            return True
        else:
            return False


@DatasetReader.register("textannotation_ner")
class TextAnnotationDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenized file where each line is in the following format:

    WORD POS-TAG CHUNK-TAG NER-TAG

    with a blank line indicating the end of each sentence
    and '-DOCSTART- -X- -X- O' indicating the end of each article,
    and converts it into a ``Dataset`` suitable for sequence tagging.

    Each ``Instance`` contains the words in the ``"tokens"`` ``TextField``.
    The values corresponding to the ``tag_label``
    values will get loaded into the ``"tags"`` ``SequenceLabelField``.

    This dataset reader ignores the "article" divisions and simply treats
    each sentence as an independent ``Instance``. (Technically the reader splits sentences
    on any combination of blank lines and "DOCSTART" tags; in particular, it does the right
    thing on well formed inputs.)

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    coding_scheme: ``str``, optional (default=``IOB1``)
        Specifies the coding scheme for ``ner_labels`` and ``chunk_labels``.
        Valid options are ``IOB1`` and ``BIOUL``.  The ``IOB1`` default maintains
        the original IOB1 scheme in the CoNLL 2003 NER data.
        In the IOB1 scheme, I is a token inside a span, O is a token outside
        a span and B is the beginning of span immediately following another
        span of the same type.
    label_namespace: ``str``, optional (default=``labels``)
        Specifies the namespace for the chosen ``tag_label``.
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 coding_scheme: str = "IOB1",
                 strategy: str = "trust_labels",
                 sentence_length_threshold: int = -1,
                 label_namespace: str = "labels",
                 labelset: List[str] = ["PER", "ORG", "LOC", "MISC"],
                 recall=1.0) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        if coding_scheme not in ("IOB1", "BIOUL", "B"):
            raise ConfigurationError("unknown coding_scheme: {}".format(coding_scheme))

        self.coding_scheme = coding_scheme
        self.label_namespace = label_namespace
        # this class reads into this scheme.
        self._original_coding_scheme = "IOB1"

        self.strategy = strategy
        self.recall = recall

        self.labelset = labelset

        self.alltags = {"O": 0}
        if self.coding_scheme == "IOB1":
            for label in self.labelset:
                self.alltags["B-" + label] = len(self.alltags)
                self.alltags["I-" + label] = len(self.alltags)
        elif self.coding_scheme == "BIOUL":
            for label in self.labelset:
                self.alltags["B-" + label] = len(self.alltags)
                self.alltags["I-" + label] = len(self.alltags)
                self.alltags["U-" + label] = len(self.alltags)
                self.alltags["L-" + label] = len(self.alltags)
        elif self.coding_scheme == "B":
            self.alltags["U-MNT"] = 1

        print("=========================================================================")
        print("     Warning: this dataset reader should be used programmatically!! ")
        print("   A bunch of weird stuff regarding how tag dictionaries are created.")
        print("=========================================================================")

        self.sentence_length_threshold = sentence_length_threshold

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        fnames = os.listdir(file_path)

        for fname in fnames:
            doc = ccg.load_document_from_json(file_path + "/" + fname)
            label_indices = ["O"] * len(doc.tokens)
            if "NER_CONLL" in doc.view_dictionary:
                ner = doc.get_ner_conll
            if "NER_ONTONOTES" in doc.view_dictionary:
                ner = doc.get_ner_ontonotes

            if ner is not None:
                if ner.cons_list is not None:

                    for cons in ner:

                        # this will randomly remove entities.
                        if random.random() > self.recall:
                            continue

                        tag = cons['label']
                        # constituent range end is one past the token
                        for i in range(cons['start'], cons['end']):
                            pref = "I-"
                            # in IOB1: you can't start a sentence with B
                            if i not in doc.sentence_end_position and i == cons["start"] \
                                    and label_indices[i-1][2:] == tag:
                                pref = "B-"

                            label_indices[i] = pref + tag

            else:
                print("doc has no ner: ", fname)

            for start, end in zip([0] + doc.sentence_end_position[:-1], doc.sentence_end_position):
                sent_toks = doc.tokens[start:end]
                ner_tags = label_indices[start:end]
                tokens = [Token(token) for token in sent_toks]

                if -1 < self.sentence_length_threshold < len(tokens):
                    logger.warning("Discarding sentence with length {}".format(len(tokens)))
                    continue

                yield self.text_to_instance(tokens, ner_tags=ner_tags)

    def text_to_instance(self,  # type: ignore
                         tokens: List[Token],
                         pos_tags: List[str] = None,
                         ner_tags: List[str] = None
                         ) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """

        # Recode the labels if necessary.
        if self.coding_scheme == "BIOUL":
            coded_ner = to_bioul(ner_tags, encoding=self._original_coding_scheme)
        elif self.coding_scheme == "B":
            # convert to binary mentions.
            coded_ner = ["O" if t == "O" else "U-MNT" for t in ner_tags]
        else:
            # the default IOB1
            coded_ner = ner_tags

        fix_coded_ner = []
        for t in coded_ner:
            if t[-1] == "-":
                fix_coded_ner.append("O")
            else:
                fix_coded_ner.append(t)
        coded_ner = fix_coded_ner
            
        # pylint: disable=arguments-differ
        sequence = TextField(tokens, self._token_indexers)
        words = [x.text for x in tokens]
        instance_fields: Dict[str, Field] = {'tokens': sequence, "metadata": MetadataField(
            {"words": words, "orig_tags": coded_ner}), "donotuse": SequenceLabelField(coded_ner, sequence,
                                                                                      label_namespace="labels")}

        tag_marginals = []
        for tag in coded_ner:
            if tag == "O":
                if self.strategy == "trust_labels":
                    # this strategy believes the tags completely
                    tag_marginal = np.zeros(len(self.alltags)) - 10000
                    tag_marginal[self.alltags[tag]] = 0
                    tag_marginals.append(ArrayField(tag_marginal))
                elif self.strategy == "uniform":
                    tag_marginal = np.zeros(len(self.alltags))
                    tag_marginals.append(ArrayField(tag_marginal))
                    # this strategy will express ignorance over all possibilities.
                else:
                    raise ConfigurationError("Unknown strategy: " + self.strategy)
            else:
                # we always fully trust the given labels.
                # this strategy believes the tags completely
                tag_marginal = np.zeros(len(self.alltags)) - 10000
                tag_marginal[self.alltags[tag]] = 0
                tag_marginals.append(ArrayField(tag_marginal))

        instance_fields['tags'] = ListField(tag_marginals)

        return Instance(instance_fields)






