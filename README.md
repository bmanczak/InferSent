# InferSent
In this repo I reproduce the sentence encoders from the paper "Supervised Learning of Universal Sentence Representations from Natural Language Inference Data" by Conneau et al. (2017) and evaluate resulting models on SentEval testing framework.

### Environment 
In ``env.yml`` please find the ``acts`` environment containing all the packages needed for training and evaluation.

### Pretrained models
Below please find the links to PyTorch lighting logs from training, including ``.ckpt`` checkpoints and ``TensorBoard`` logs.

[Avg Glove word embeddings]()

[Uni-directional LSTM]()

[Bi-directional LSTM]()

[Bi-directional LSTM with max pooling]()

### Structure 

```
InferSent
| env.yml
| models.py   % includes the definition of the encoders
| dataset.py  % fetches the dataset related objects from torchtext
| task.py     % contains PyTorchLightining trainer
| train.py    % training logic
| utils.py    % utiltiy functions
| SentEval
|   ...       % scripts from SentEval repo

```
