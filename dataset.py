#Torchtext 
import torchtext
from torchtext.datasets import SNLI
from torchtext.data import Field, BucketIterator

from utils import tokenizer

def create_iterators(batch_size = 64, return_label = True):
    """
    Createas the SNLI data iterator.
    """

    TEXT = Field(sequential=True, use_vocab=True, tokenize=tokenizer, lower=True, include_lengths=True)
    LABEL = Field(sequential = False, use_vocab= True)

    train, val, test = SNLI.splits(text_field=TEXT, label_field=LABEL)

    TEXT.build_vocab(train, vectors= torchtext.vocab.GloVe(name='840B'))
    LABEL.build_vocab(train)

    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train, val, test),
        batch_size=batch_size,
        shuffle = True)

    if return_label:
        return TEXT, LABEL, train_iter, val_iter, test_iter
    else:
        return TEXT,train_iter, val_iter, test_iter