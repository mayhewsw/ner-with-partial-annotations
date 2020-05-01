from typing import List

import torch
import torch.optim as optim
from allennlp.common.file_utils import cached_path
from allennlp.data import Vocabulary, Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer, PretrainedBertIndexer
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder, PretrainedBertEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.nn import Activation

from mylib.models.marginal_crf_tagger import MarginalCrfTagger
from mylib.dataset_readers.textannotation_ner import TextAnnotationDatasetReader
import numpy as np


# these are the gold b values for the training datasets (see Table 1 in the paper)
goldb = {"eng": 0.166, "esp": 0.123, "deu": 0.08, "ned": 0.095, "amh": 0.112, "ara": 0.126, "hin": 0.074, "som": 0.112, "ben": 0.12, "uig": 0.1}


# Set these to the desired values!
USING_BERT = False
DATA_DIR = "data/"

def get_model(pretrained_file: str, WORD_EMB_DIM: int, vocab: Vocabulary, num_tags: int):
    """
    This creates a new model and returns it along with some other variables.
    :param pretrained_file:
    :param WORD_EMB_DIM:
    :param vocab:
    :param num_tags:
    :return:
    """

    CNN_EMB_DIM = 128
    CHAR_EMB_DIM = 16

    weight = _read_pretrained_embeddings_file(pretrained_file, WORD_EMB_DIM, vocab, "tokens")
    token_embedding = Embedding(num_embeddings=weight.shape[0], embedding_dim=weight.shape[1], weight=weight, vocab_namespace="tokens")
    char_embedding = Embedding(num_embeddings=vocab.get_vocab_size("token_characters"),
                               embedding_dim=CHAR_EMB_DIM, vocab_namespace="token_characters")

    char_encoder = CnnEncoder(embedding_dim=CHAR_EMB_DIM, num_filters=CNN_EMB_DIM, ngram_filter_sizes=[3],
                              conv_layer_activation=Activation.by_name("relu")())
    token_characters_embedding = TokenCharactersEncoder(embedding=char_embedding, encoder=char_encoder)

    if USING_BERT:
        print("USING BERT EMBEDDINGS")
        bert_emb = PretrainedBertEmbedder("bert-base-multilingual-cased")
        tfe = BasicTextFieldEmbedder({"bert": bert_emb, "token_characters": token_characters_embedding},
                                     embedder_to_indexer_map={"bert": ["bert", "bert-offsets"],
                                                              "token_characters": ["token_characters"]},
                                     allow_unmatched_keys=True)

        EMBEDDING_DIM = CNN_EMB_DIM + 768
    else:
        EMBEDDING_DIM = CNN_EMB_DIM + WORD_EMB_DIM
        tfe = BasicTextFieldEmbedder({"tokens": token_embedding, "token_characters": token_characters_embedding})

    HIDDEN_DIM = 256

    encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True,
                                                  bidirectional=True, dropout=0.5, num_layers=2))

    model = MarginalCrfTagger(vocab, tfe, encoder, num_tags, include_start_end_transitions=False,
                              calculate_span_f1=True, dropout=0.5, label_encoding="BIOUL", constrain_crf_decoding=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if torch.cuda.is_available():
        print("Using GPU")
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    return model, optimizer, cuda_device


def get_stats(dataset: List[Instance]):
    """
    Given a list of instances, count certain statistics.
    :param dataset: just a list of instances.
    :return: dictionary of statistics with descriptive names
    """
    total_toks = 0
    total_hard_tags = 0
    total_tags = 0

    for inst in dataset:
        orig_words = inst["metadata"]["words"]
        orig_tags = inst["metadata"]["orig_tags"]

        for k in range(len(orig_words)):

            # get the number that were originall given to us.
            if orig_tags[k] != "O":
                total_tags += 1

            # O is always the first index
            a = np.exp(inst["tags"][k].array)
            if max(a) != a[0]:
                total_hard_tags += 1
            total_toks += 1

    return {"total_toks": total_toks, "total_tags": total_tags,
            "total_hard_tags" : total_hard_tags,
            "ratio": total_tags / float(total_toks)}


def get_data(lang: str, recall: int = 1.0):

    if lang == "eng":
        # https://nlp.stanford.edu/projects/glove/
        pretrained_file = "/path/to/embeddings/glove.6B.50d.txt"
        WORD_EMB_DIM = 50
        lower_tokens = True
        labels = ["PER", "ORG", "LOC", "MISC"]
    elif lang == "esp":
        # https://github.com/glample/tagger/issues/44
        pretrained_file = "/path/to/embeddings/esp64"
        WORD_EMB_DIM = 64
        lower_tokens = False
        labels = ["PER", "ORG", "LOC", "MISC"]
    elif lang == "ned":
        # https://github.com/glample/tagger/issues/44
        pretrained_file = "/path/to/embeddings/ned64"
        WORD_EMB_DIM = 64
        lower_tokens = False
        labels = ["PER", "ORG", "LOC", "MISC"]
    elif lang == "deu":
        # https://github.com/glample/tagger/issues/44
        pretrained_file = "/path/to/embeddings/deu64"
        WORD_EMB_DIM = 64
        lower_tokens = False
        labels = ["PER", "ORG", "LOC", "MISC"]
    else:
        print("Other languages not supported in this release!")

    indexers = {"tokens": SingleIdTokenIndexer(lowercase_tokens=lower_tokens),
                "token_characters": TokenCharactersIndexer(min_padding_length=3),
                }

    if USING_BERT:
        indexers["bert"] = PretrainedBertIndexer("bert-base-multilingual-cased", do_lowercase=False)

    reader = TextAnnotationDatasetReader(coding_scheme="BIOUL", token_indexers=indexers,
                                         strategy="trust_labels", recall=recall, labelset=labels)
    # Important that validation_reader has strategy="trust_labels" because otherwise the gold labels would be wrong.
    # recall is always 1.0 for validation_reader
    validation_reader = TextAnnotationDatasetReader(coding_scheme="BIOUL", token_indexers=indexers,
                                                    strategy="trust_labels", labelset=labels)

    if recall == 1:
        train_dataset = reader.read(cached_path(DATA_DIR + "{}/Trainp0.9r0.5".format(lang)))
    else:
        train_dataset = reader.read(cached_path(DATA_DIR + "{}/Train".format(lang)))
    validation_dataset = validation_reader.read(cached_path(DATA_DIR + "{}/Dev".format(lang)))
    test_dataset = validation_reader.read(cached_path(DATA_DIR + "{}/Test".format(lang)))

    all_insts = train_dataset + validation_dataset + test_dataset
    vocab = Vocabulary.from_instances(all_insts)

    # The following is a hack because we need
    del vocab._token_to_index["labels"]
    del vocab._index_to_token["labels"]
    for k in reader.alltags:
        vocab.add_token_to_namespace(k, "labels")
    print(vocab.get_token_to_index_vocabulary("labels"))

    return {"train": train_dataset, "dev": validation_dataset, "test": test_dataset, "reader": reader,
            "WORD_EMB_DIM": WORD_EMB_DIM, "pretrained_file": pretrained_file, "vocab": vocab}


def get_data_binary(lang: str, recall: int = 1.0):

    labels = ["MNT"]
    coding_scheme = "B"

    # TODO: these are either Glove embeddings, or the embeddings from Lample et al.
    if lang == "eng":
        pretrained_file = "/path/to/embeddings/glove.6B.50d.txt"
        WORD_EMB_DIM = 50
        lower_tokens = True
    elif lang == "esp":
        pretrained_file = "/path/to/embeddings/esp64"
        WORD_EMB_DIM = 64
        lower_tokens = False
    elif lang == "ned":
        pretrained_file = "/path/to/embeddings/ned64"
        WORD_EMB_DIM = 64
        lower_tokens = False
    elif lang == "deu":
        pretrained_file = "/path/to/embeddings/deu64"
        WORD_EMB_DIM = 64
        lower_tokens = False
    else:
        print("Other languages not supported in this release!")

    indexers = {"tokens": SingleIdTokenIndexer(lowercase_tokens=lower_tokens),
                "token_characters": TokenCharactersIndexer(min_padding_length=3),
                }

    using_bert = False
    if using_bert:
        indexers["bert"] = PretrainedBertIndexer("bert-base-multilingual-cased", do_lowercase=False)

    reader = TextAnnotationDatasetReader(coding_scheme=coding_scheme, token_indexers=indexers,
                                         strategy="trust_labels", recall=recall, labelset=labels)
    # Important that validation_reader has strategy="trust_labels" because otherwise the gold labels would be wrong.
    # recall is always 1.0 for validation_reader
    validation_reader = TextAnnotationDatasetReader(coding_scheme=coding_scheme, token_indexers=indexers,
                                                    strategy="trust_labels", labelset=labels)

    if recall == 1:
        train_dataset = reader.read(cached_path(DATA_DIR + "{}/Trainp0.9r0.5".format(lang)))
    else:
        train_dataset = reader.read(cached_path(DATA_DIR + "{}/Train".format(lang)))
    validation_dataset = validation_reader.read(cached_path(DATA_DIR + "{}/Dev".format(lang)))
    test_dataset = validation_reader.read(cached_path(DATA_DIR + "{}/Test".format(lang)))

    all_insts = train_dataset + validation_dataset + test_dataset
    vocab = Vocabulary.from_instances(all_insts)

    # The following is a hack
    del vocab._token_to_index["labels"]
    del vocab._index_to_token["labels"]

    for k in reader.alltags:
        vocab.add_token_to_namespace(k, "labels")
    print(vocab.get_token_to_index_vocabulary("labels"))

    return {"train": train_dataset, "dev": validation_dataset, "test": test_dataset, "reader": reader,
            "WORD_EMB_DIM": WORD_EMB_DIM, "pretrained_file": pretrained_file, "vocab": vocab}


def get_b(data: List[Instance]):
    total_toks = 0
    total_entity_mass = 0

    for inst in data:
        orig_words = inst["metadata"]["words"]

        for k in range(len(orig_words)):
            a = np.exp(inst["tags"][k].array)
            # O is always the first index.
            total_entity_mass += sum(a) - a[0]
            total_toks += 1

    return total_entity_mass / total_toks


def get_hard_b(data: List[Instance]):
    total_toks = 0
    total_tags = 0

    for inst in data:
        orig_words = inst["metadata"]["words"]

        for k in range(len(orig_words)):
            a = np.exp(inst["tags"][k].array)
            # O is always the first index.
            if max(a) != a[0]:
                total_tags += 1
            total_toks += 1

    return total_tags / float(total_toks)


def correct_and_relabel(data: List[Instance], desiredb: float):
    """
    This takes data with predictions on it.
    We will modify the predictions so that the bvalue is currentb.
    we order positive predictions by probability.
    If something is predicted as being positive, we order them by probability and select the best.
    We also set all the rest of the positive predictions to 50/50.
    If something is predicted as being negative, it stays exactly as is.
    :param data:
    :param desiredb:
    :return:
    """

    # There is no batching in this function.

    # currentb = get_b(data)
    data_stats = get_stats(data)
    total_toks = data_stats["total_toks"]
    total_tags = data_stats["total_hard_tags"]
    desired_num_tags = int(desiredb * total_toks)

    print("we have {} tags, but we want {} tags".format(total_tags, desired_num_tags))

    positive_ids = []

    for i, inst in enumerate(data):
        # getting this for the seq len
        seq_len = len(inst["metadata"]["words"])
        for k in range(seq_len):
            # these are always in the log space.
            # exponentiated so they are probabilities.
            marginals = np.exp(inst["tags"][k].array)
            assert len(marginals) == 2

            id = "{}-{}".format(i, k)
            positive_ids.append((marginals[1], id))

    positive_ids = sorted(positive_ids, reverse=True)[:desired_num_tags]

    # create a set that contains only the ids for easier lookup
    positive_ids = set([pi[1] for pi in positive_ids])

    for i, inst in enumerate(data):
        # getting this for the seq len
        seq_len = len(inst["metadata"]["words"])
        for k in range(seq_len):
            # these are always in the log space.
            # exponentiated so they are probabilities.
            marginals = np.exp(inst["tags"][k].array)

            # if you are in my list, then tagged as B-MNT
            id = "{}-{}".format(i, k)
            if id in positive_ids:
                # tag definitively as mention.
                # this value matches what we gave in a different file.
                # is that important, I'm not sure.
                marginals[0] = -10000
                marginals[1] = 0
            else:
                # for positive predictions that didn't make the cut, this forces them to be equal
                # for negative predictions, nothing changes.
                new_pos_value = min(marginals[1], 0.5)
                marginals[0] = np.log(1 - new_pos_value)
                marginals[1] = np.log(new_pos_value)

            inst["tags"][k].array = marginals

    # I think we don't need to return the data because it is being modified in place. riiight?


def copy_weights(data: List[Instance], binary_data: List[Instance]):
    # here we have data with some labels in the form of distributions over labelsets.
    # except the labelset distribution is binary, yes/no.

    assert len(data) == len(binary_data)

    for inst, binary_inst in zip(data, binary_data):
        seq_len = len(inst["metadata"]["words"])
        orig_tags = inst["metadata"]["orig_tags"]

        for k in range(seq_len):
            # we are confident in the positive labels in data
            # just not the negative labels.
            if orig_tags[k] == "O":

                num_tags = len(inst["tags"][k].array)

                # this is a 2 element probability distribution
                marginals = np.exp(binary_inst["tags"][k].array)

                # Our binary model has predicted that this word is a
                # positive element.
                if marginals[1] > marginals[0]:
                    new_marginals = np.log(np.ones(num_tags) / float(num_tags))
                else:
                    new_marginals = np.ones(num_tags)
                    # O gets the same confidence
                    new_marginals[0] = np.log(marginals[0])
                    # then split whatever's left among the remaining tokens.
                    remainder = 1 - marginals[0]
                    for j in range(1, num_tags):
                        new_marginals[j] = np.log(remainder / float(num_tags-1))

                # we are trying to set this value
                inst["tags"][k].array = new_marginals


def dump_dataset(data: List[Instance], outfile: str):
    print("Writing to ", outfile)
    with open(outfile, "w") as out:
        for inst in data:
            words = inst["metadata"]["words"]
            orig_tags = inst["metadata"]["orig_tags"]

            for k in range(len(words)):
                scores = " ".join(['{:.2f}'.format(n) for n in np.exp(inst["tags"][k].array)])
                out.write("{} {} {}\n".format(words[k], orig_tags[k], scores))

            out.write("\n")
