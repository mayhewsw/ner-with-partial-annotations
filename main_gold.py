import os
import random
import shutil
import sys

import numpy as np
import torch
from allennlp.common.util import dump_metrics, prepare_global_logging
from allennlp.data.iterators import BasicIterator
from allennlp.training import util
from allennlp.training.trainer import Trainer

from mylib.util import get_model, get_stats, get_data

seed = 2233235
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

#  CONFIG SECTION
lang = sys.argv[1]
expname = "exp1_gold"
batch_size = 32


# A bunch of parameters from the data.
# this recall thing is a hack so it selects the gold data.
data_dict = get_data(lang, recall=0.999)
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
serialization_dir = "/scratch/mayhew/partial-ner-allan/{}/{}".format(lang, expname)
if os.path.exists(serialization_dir):
    print("serialization directory exists... ")
    r = input("Serialization dir {} exists. Remove? y/n  ".format(serialization_dir))
    if r == "y":
        shutil.rmtree(serialization_dir)
    else:
        print("Not removing directory")
        sys.exit()


iterator = BasicIterator(batch_size=batch_size)
iterator.index_with(vocab)

model, optimizer, cuda_device = get_model(pretrained_file, WORD_EMB_DIM, vocab, len(reader.alltags))

prepare_global_logging(serialization_dir, False)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=75,
                  validation_metric="+f1-measure-overall",
                  cuda_device=cuda_device,
                  num_serialized_models_to_keep=3,
                  serialization_dir=serialization_dir)

metrics = trainer.train()

test_metrics = util.evaluate(trainer.model, test_dataset, iterator,
                             cuda_device=trainer._cuda_devices[0],  # pylint: disable=protected-access,
                             batch_weight_key="")

for key, value in test_metrics.items():
    metrics["test_" + key] = value

dump_metrics(os.path.join(serialization_dir, "metrics.json"), metrics, log=True)
