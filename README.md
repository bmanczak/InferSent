# InferSent
In this repo I reproduce the sentence encoders from the paper "Supervised Learning of Universal Sentence Representations from Natural Language Inference Data" by Conneau et al. (2017) and evaluate resulting models on SentEval testing framework.

### Environment 
In ``env.yml`` please find the ``acts`` environment containing all the packages needed for training and evaluation.

### Pretrained models
All pretrained models are avalaiable [here](https://drive.google.com/drive/folders/1OJ8vpZthCWl77e-BRqemUATmQO5qmp74?usp=sharing).

### Structure 

```
InferSent
| env.yml
| models.py    % includes the definition of the encoders
| dataset.py   % fetches the dataset related objects from torchtext
| task.py      % contains PyTorchLightining trainer
| train.py     % training logic
| utils.py     % utiltiy functions
| demoNotebook.ipnyb
| SentEval
|   ...       % scripts from SentEval repo

```
