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

from mylib.util import get_model, get_stats, get_data

seed = 2233235
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

#  CONFIG SECTION
lang = sys.argv[1]
expname = "exp1_long"
batch_size = 32

# how many overall iterations.
iterations = 10

# A bunch of parameters from the data.
data_dict = get_data(lang)
train_dataset = data_dict["train"]
validation_dataset = data_dict["dev"]
test_dataset = data_dict["test"]
reader = data_dict["reader"]
WORD_EMB_DIM = data_dict["WORD_EMB_DIM"]
pretrained_file = data_dict["pretrained_file"]
vocab = data_dict["vocab"]

print("Train data stats:")
stats = get_stats(train_dataset)
print("total toks: ", stats["total_toks"])
print("total tags: ", stats["total_tags"])

# don't put a slash after this.
serialization_dir = "/scratch/models/{}/{}".format(lang, expname)
if os.path.exists(serialization_dir):
    print("serialization directory exists... ")
    r = input("Serialization dir {} exists. Remove? y/n  ".format(serialization_dir))
    if r == "y":
        shutil.rmtree(serialization_dir)
    else:
        print("Not removing directory")
        sys.exit()

ltd = len(train_dataset)
folds = [train_dataset[:ltd//2], train_dataset[ltd//2:]]

# FIXME: their code trains on each fold before updating either one. Consider doing this also...
for big_iter in range(iterations):
    print(" ---- BIG ITERATION: {} ---- ".format(big_iter))
    for i, fold in enumerate(folds):

        model,optimizer, cuda_device = get_model(pretrained_file, WORD_EMB_DIM, vocab, len(reader.alltags))

        iterator = BasicIterator(batch_size=batch_size)
        iterator.index_with(vocab)

        ser_dir_iter = serialization_dir + "/iter-{}-{}".format(big_iter, i)
        prepare_global_logging(ser_dir_iter, False)

        trainer = Trainer(model=model,
                          optimizer=optimizer,
                          iterator=iterator,
                          train_dataset=fold,
                          validation_dataset=validation_dataset,
                          patience=10,
                          num_epochs=45,
                          validation_metric="+f1-measure-overall",
                          cuda_device=cuda_device,
                          num_serialized_models_to_keep=3,
                          serialization_dir=ser_dir_iter)

        metrics = trainer.train()

        print("Using model trained on fold {} to tag fold {}.".format(i, 1-i))
        insts = folds[1-i]
        for inst in tqdm(insts):
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


print(" ---- Finished with training folds, now training all together ---- ")
iterator = BasicIterator(batch_size=batch_size)
iterator.index_with(vocab)

model, optimizer, cuda_device = get_model(pretrained_file, WORD_EMB_DIM, vocab, len(reader.alltags))

ser_dir_iter = serialization_dir + "/final"
prepare_global_logging(ser_dir_iter, False)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=folds[0] + folds[1],
                  validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=45,
                  validation_metric="+f1-measure-overall",
                  cuda_device=cuda_device,
                  num_serialized_models_to_keep=3,
                  serialization_dir=ser_dir_iter)

trainer.train()

test_metrics = util.evaluate(trainer.model, test_dataset, iterator,
                             cuda_device=trainer._cuda_devices[0],  # pylint: disable=protected-access,
                             batch_weight_key="")

for key, value in test_metrics.items():
    metrics["test_" + key] = value

dump_metrics(os.path.join(ser_dir_iter, "metrics.json"), metrics, log=True)
