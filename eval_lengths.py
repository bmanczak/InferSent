import os
#import spacy
import pytorch_lightning as pl
import numpy as np
import argparse
from easydict import EasyDict as edict
from collections import Counter

import torch
import torch.nn as nn

import torchtext
from torchtext.data import Field
from torchtext.datasets import SNLI

from train import InferSent
from dataset import create_iterators
from task import InferSent as InferSentPool
from taskRelu import InferSent as InferSent

TEXT, LABEL, train_iter, val_iter, test_iter = create_iterators(batch_size = 4, return_label = True) # feel free to change the batch size to your computer's capabilities 

def masked_accuracy(model, test_iter, threshold, length_source = "premise", comparison = "smaller"):
    """Calculates the accuracy for examples with lenght greater/smaller than threshold."""

    acc = 0
    count = 0

    for batch in test_iter:

        if length_source == "premise":
            if comparison == "smaller":
                mask = batch.premise[1] <= threshold
            else:
                mask = batch.premise[1] >= threshold
        else:
            if comparison == "smaller":
                mask = batch.hypothesis[1] <= threshold
            else:
                mask = batch.hypothesis[1] >= threshold

        if any(mask):
            batch = edict({"premise": (batch.premise[0][:, mask], batch.premise[1][mask]),
                          "hypothesis": ( batch.hypothesis[0][:, mask], batch.hypothesis[1][mask] ),
                          "label": batch.label[mask]})

            count += sum(mask)

            preds = model(batch).argmax(dim = -1)
            labels = batch.label - 1

            acc += (preds == labels).float().sum()

        #if count > 1000:
        #    break
        
    #print("Accuracy:", (acc/count *100).item(), "%")
    return (acc/count).item()    

if __name__ == '__main__':

    awe = InferSent.load_from_checkpoint("../saved_models/word_embs/lightning_logs/version_7594034/checkpoints/epoch=7-step=68671.ckpt")
    uniLstm = InferSent.load_from_checkpoint("../saved_models/uniLSTM/lightning_logs/version_7593109/checkpoints/epoch=12-step=111591_8152.ckpt")
    biLstm = InferSent.load_from_checkpoint("../saved_models/biLSTM/lightning_logs/version_7593123/checkpoints/epoch=10-step=94423_806.ckpt")
    biLSTMPool = InferSentPool.load_from_checkpoint("../saved_models/biLSTMPool/lightning_logs/version_7594584/checkpoints/epoch=6-step=60087.ckpt")

    models = [awe, uniLstm, biLstm, biLSTMPool]
    values = [8,22, 5, 12] # 10th percentile premise
    length_source = ["premise", "premise", "hypothesis", "hypothesis"]
    comparison = ["smaller", "larger", "smaller", "largerr"]


    for model in models:
        try:
            print("Running for model", model, flush = True)
            for threshold,source, comp in zip(values, length_source, comparison):
                print(f"Accuracy for parameters threshold={threshold},source={source}, comparison={comp}", flush = True)
                print(masked_accuracy(model, test_iter, threshold, source, comp))
            
            print("\n", flush = True)
            print("-"*50, flush = True)
            print("\n", flush=True)
        
        except Exception as e:
            print("Exception", e)