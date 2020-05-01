import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F
from allennlp.common.util import dump_metrics, prepare_global_logging
from allennlp.data.iterators import BasicIterator
from allennlp.training import util
from allennlp.training.trainer import Trainer
from tqdm import tqdm

from mylib.util import get_model, get_stats, get_data, get_b, correct_and_relabel, get_data_binary, get_hard_b, copy_weights
from mylib.util import dump_dataset, goldb
from mylib.weights import add_weights

seed = 2233235
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

#  CONFIG SECTION
lang = sys.argv[1]
expname = "cbl"
batch_size = 32

# A bunch of parameters from the data.
data_dict = get_data_binary(lang)
train_dataset = data_dict["train"]
validation_dataset = data_dict["dev"]
test_dataset = data_dict["test"]
reader = data_dict["reader"]
WORD_EMB_DIM = data_dict["WORD_EMB_DIM"]
pretrained_file = data_dict["pretrained_file"]
vocab = data_dict["vocab"]

bvalue = get_b(train_dataset)
target_bvalue = goldb[lang] + 0.05
delta = 0.02

init_with_weights = False
if init_with_weights:
    print("======================== INIT WITH WEIGHTS ==========================")
    # add weights to file.
    target_bvalue = goldb[lang]
    add_weights(train_dataset, target_bvalue)
    expname = "cbl_combined"


print("Train data stats:")
stats = get_stats(train_dataset)
print("total toks: ", stats["total_toks"])
print("total tags: ", stats["total_tags"])

# don't put a slash after this.
serialization_dir = "/scratch/models/{}/{}".format(lang, expname)
if os.path.exists(serialization_dir):
    print("serialization directory exists... ")
    if "deleteme" in serialization_dir:
        shutil.rmtree(serialization_dir)
    else:
        r = input("Serialization dir {} exists. Remove? y/n  ".format(serialization_dir))
        if r == "y":
            shutil.rmtree(serialization_dir)
        else:
            print("Not removing directory")
            sys.exit()

brange = []
currb = bvalue
while currb < target_bvalue:
    brange.append(currb)
    currb += delta

print(brange)

for iteration, currentb in enumerate(brange):
    print(" ---- ITERATION: {} ---- ".format(currentb))

    # get a new model for each iteration.
    model, optimizer, cuda_device = get_model(pretrained_file, WORD_EMB_DIM, vocab, len(reader.alltags))

    iterator = BasicIterator(batch_size=batch_size)
    iterator.index_with(vocab)

    ser_dir_iter = serialization_dir + "/iter-{}".format(iteration)
    prepare_global_logging(ser_dir_iter, False)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      patience=10,
                      num_epochs=25,   # FIXME: consider more iterations.
                      validation_metric="+f1-measure-overall",
                      cuda_device=cuda_device,
                      num_serialized_models_to_keep=3,
                      serialization_dir=ser_dir_iter)

    metrics = trainer.train()

    print("tagging training data...")
    for inst in tqdm(train_dataset):
        model.eval()
        output = model.forward_on_instance(inst)
        seq_len, num_tags = output["logits"].shape

        orig_tags = inst["metadata"]["orig_tags"]

        for k in range(seq_len):
            # orig_tags keeps track of the original labels on the words.
            # if an item has a label, then we trust it completely. If not, then we relabel it.
            if orig_tags[k] == "O":
                # marginals are always understood to be in log space
                inst["tags"][k].array = F.log_softmax(torch.FloatTensor(output["logits"][k]), dim=-1)

    print("hard b value", get_hard_b(train_dataset))
    correct_and_relabel(train_dataset, currentb)
    print("hard b value", get_hard_b(train_dataset))

print(" ---- Finished CBL training, now training all together ---- ")

# heh heh silly names for datasets.
binary_train_dataset = train_dataset


# A bunch of parameters from the data.
data_dict = get_data(lang)
train_dataset = data_dict["train"]
validation_dataset = data_dict["dev"]
test_dataset = data_dict["test"]
reader = data_dict["reader"]
WORD_EMB_DIM = data_dict["WORD_EMB_DIM"]
pretrained_file = data_dict["pretrained_file"]
vocab = data_dict["vocab"]

train_dataset = sorted(train_dataset, key=lambda i: i["metadata"]["words"])
binary_train_dataset = sorted(binary_train_dataset, key=lambda i: i["metadata"]["words"])

copy_weights(train_dataset, binary_train_dataset)

print(get_b(train_dataset))
print(get_hard_b(train_dataset))

dump_dataset(train_dataset, "final_train_{}.txt".format(lang))

iterator = BasicIterator(batch_size=batch_size)
iterator.index_with(vocab)

model, optimizer, cuda_device = get_model(pretrained_file, WORD_EMB_DIM, vocab, len(reader.alltags))

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=45,
                  validation_metric="+f1-measure-overall",
                  cuda_device=cuda_device,
                  num_serialized_models_to_keep=3,
                  serialization_dir=serialization_dir + "/final")

trainer.train()

test_metrics = util.evaluate(trainer.model, test_dataset, iterator,
                             cuda_device=trainer._cuda_devices[0], # pylint: disable=protected-access,
                             batch_weight_key="")

for key, value in test_metrics.items():
    metrics["test_" + key] = value

print(serialization_dir)
dump_metrics(os.path.join(serialization_dir + "/final", "metrics.json"), metrics, log=True)
