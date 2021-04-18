# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging
from collections import Counter

PATH_TO_SENTEVAL = "SentEval/"
# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
from task import InferSent
import senteval
import torch.nn as nn
import torch
import spacy
import torchtext
import argparse



def build_vocab(params):
    embeddings = []
    ids = sorted(params.word2id.values())
    for idx in ids:
        word = params.id2word[idx]
        if word in params.word_vec:
            embeddings.append([params.word_vec[word]])
    embeddings = np.array(embeddings)
    print("EMBEDDINGS SHAPE",embeddings.shape)
    return torch.from_numpy(embeddings).squeeze(dim = 1) # [voacb_size, hidden_dim]

def tokenizer(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

def create_counter(sentences):
    words = []
    for s in sentences:
        s = " ".join(s)
        tokens = tokenizer(s)
        for elem in tokens:
            words.append(elem)
    counter = Counter(words)
    return counter
    
# SentEval prepare and batcher
def prepare(params, samples):
    counter = create_counter(samples)
    params.embeddings = torchtext.vocab.Vocab(counter = counter, vectors = torchtext.vocab.GloVe(name='840B'))
    return

def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []
  
    embedding_layer = nn.Embedding.from_pretrained(params.embeddings.vectors, freeze= True)                           
    model.embedding = embedding_layer

    for sent in batch:
        sentvec = []
        length = 0
        for word in sent:
            #if word in params.word_vec:
                #sentvec.append(params.word2id[word])
            sentvec.append(params.embeddings.stoi[word])
            length += 1
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)
        try:
            sentvec = model(torch.LongTensor(sentvec).unsqueeze(dim = 1), torch.LongTensor([length]) )
        except:  # dirty, neeeded for word embedding model
            sentvec = model(torch.LongTensor(sentvec).unsqueeze(dim = 1))
        #sentvec = np.mean(sentvec, 0)

        embeddings.append(sentvec.detach().numpy())

    embeddings = np.vstack(embeddings)
    return embeddings



if __name__ == "__main__":

    # Set up logger
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
    PATH_TO_DATA = 'SentEval/data'

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument("--model_path", help = "Path to InferSent model")

    parser.add_argument("--usepytorch", type = int, default= 1,
                        help = "Whether to use PyTorch for SentEval")
    
    args = parser.parse_args()

     # Set params for SentEval
    params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': args.usepytorch, 'kfold': 10}
    params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                    'tenacity': 5, 'epoch_size': 4}

        # Set PATHs
    #PATH_TO_SENTEVAL = '../'
    
    #PATH_TO_MODEL = a
    #PATH_TO_MODEL = "/Users/blazejmanczak/Desktop/School/Year1/Block5/acts/Practical/saved_models/word_embs642.ckpt"

    spacy_eng = spacy.load('en_core_web_sm')
    model = InferSent.load_from_checkpoint(args.model_path).model
    

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    """
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    """
    '''transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']'''

    #transfer_tasks = ['SST2']#, 'STS14']

    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC',
                      'MRPC', 'SICKEntailment', "SICKRelatedness"]
    results = se.eval(transfer_tasks)
    print(results)
