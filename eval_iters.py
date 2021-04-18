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

TEXT, train_iter, val_iter, test_iter = create_iterators(batch_size = 128, return_label = False) # feel free to change the batch size to your computer's capabilities 

def evaluate_SNLI(model, iterator):
    
    acc = 0
    count = 0 # of examples
    
    for batch in iterator:
        preds = model(batch)
        labels = batch.label - 1
        acc += (preds.argmax(dim=-1) == labels).float().sum().item()#.mean()
        count += batch.premise[1].shape[0] # add the batch size
        
    return acc/count

if __name__ == '__main__':

    awe = InferSent.load_from_checkpoint("../saved_models/word_embs/lightning_logs/version_7594034/checkpoints/epoch=7-step=68671.ckpt")
    uniLstm = InferSent.load_from_checkpoint("../saved_models/uniLSTM/lightning_logs/version_7593109/checkpoints/epoch=12-step=111591_8152.ckpt")
    biLstm = InferSent.load_from_checkpoint("../saved_models/biLSTM/lightning_logs/version_7593123/checkpoints/epoch=10-step=94423_806.ckpt")
    biLSTMPool = InferSentPool.load_from_checkpoint("../saved_models/biLSTMPool/lightning_logs/version_7594584/checkpoints/epoch=6-step=60087.ckpt")

    models = [awe, uniLstm, biLstm, biLSTMPool]


    for model in models:
        try:
            print(model)
            print("Validation accuracy: ", evaluate_SNLI(model, val_iter), flush = True) 
            print("Test accuracy:", evaluate_SNLI(model, test_iter), flush = True) 
        
        except Exception as e:
            print("Exception", e)