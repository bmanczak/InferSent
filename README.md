# InferSent
In this repo I reproduce the sentence encoders from the paper "Supervised Learning of Universal Sentence Representations from Natural Language Inference Data" by Conneau et al. (2017) and evaluate resulting models on SentEval testing framework. Please find summary of the findingsr in Report.pdf.

### Environment 
In ``env.yml`` please find the ``acts`` environment containing all the packages needed for training and evaluation.

### Pretrained models
All pretrained models are avalaiable [here](https://drive.google.com/drive/folders/1OJ8vpZthCWl77e-BRqemUATmQO5qmp74?usp=sharing).

### Structure 

```
InferSent
| env.yml
| models.py       % includes the definition of the encoders
| dataset.py      % fetches the dataset related objects from torchtext
| task.py         % contains PyTorchLightining trainer InferSent
| train.py        % training logic
| utils.py        % utiltiy functions
| demoNotebook.ipnyb

| eval_iters.py   % evaluate the models on SNLI 
| eval.py         % evalaue the models on SentEval

| SentEval
|   ...       % content of the SentEval repo

```

### Training
To train the model, run:

`python train.py --args`

See `train.py` for supported arguments.

### Evaluation

#### Evaluate on SNLI
Run:

`python eval_iters.py`

In this case the path to the models need to be changed in the `.py` script.


#### Evaluate on SentEval
To evaluate on SentEval, copy the [SentEval repo](https://github.com/facebookresearch/SentEval) into InferSent and then run:

`python eval.py --model_path --usepytorch`

#### Evaluate the Short/long sentences on SNLI
Run:

`python eval_lengths.py`

In this case the path to the models need to be changed in the `.py` script.
