## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from utils import create_model

class InferSent(pl.LightningModule):
    
    def __init__(self, model_name, model_hparams, optimizer_name = "SGD", optimizer_hparams = 
                                                                     {"lr": 0.1}):
        """
        Inputs:
        
        model_name: str
            Name of the model to be used. Supported ["word_embs", "lstm"]
        """
        
        super().__init__()
     
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = create_model(model_name, model_hparams)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()

        if model_name == "word_embs":
            self.dense = nn.Linear(4*300, 512)
        else: # *4 because we concatenate the sentence vectors in 4 ways
            self.dense = nn.Linear(4*self.model.lstm_hidden_size*self.model.num_directions, 512)
        self.classify = nn.Linear(512, 3) # 3 is the number of classes
        
    
    def forward(self, batch):
        x1, input_lengths_1 = batch.premise
        x2, input_lengths_2 = batch.hypothesis
        
        if self.hparams.model_name == "word_embs":
            out_premise = self.model(x1)
            out_hyp = self.model(x2)
            
        else:
            out_premise = self.model(x1, input_lengths_1)
            #print("Out premise shaep", out_premise.shape)
            out_hyp = self.model(x2, input_lengths_2)
            
        concat_out = torch.cat((out_premise, out_hyp,
                               torch.abs(out_premise - out_hyp), out_premise*out_hyp), dim = 1)
        x = F.relu(concat_out)
        x = self.dense(x)
        x = F.relu(x)
        out = self.classify(x)
        
        return out
    
    def configure_optimizers(self):
        
      if self.hparams.optimizer_name == "SGD":
          optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
          # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
          scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode='max',
                                                        factor=0.2,
                                                        patience=1,
                                                        min_lr=1e-5)

          scheduler2 = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        
          #return {"optimizer": optimizer, "lr_scheduler": scheduler1, "monitor": "val_acc"}
          return [optimizer], [{"scheduler": scheduler1, "interval":"epoch", "monitor":"val_acc"},
                                {"scheduler": scheduler2, "interval":"epoch"}] #{"lr_scheduler": scheduler1, "monitor": "val_acc"}

      elif self.hparams.optimizer_name=="Adam": # for debugging
          optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
          return optimizer

    
    def training_step(self, batch, batch_idx):
        
        preds = self(batch)
        labels = batch.label - 1
        
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        
        self.log('train_acc', acc, on_step=True, on_epoch=True) # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_loss', loss)
        
        return loss # Return tensor to call ".backward" on
    
    def validation_step(self, batch, batch_idx):
        preds = self(batch)
        labels = batch.label - 1
        
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        
        self.log("val_loss", loss, on_step = True)
        self.log('val_acc', acc) # By default logs it per epoch (weighted average over batches)
        
    def test_step(self, batch, batch_idx):
        preds = self(batch)
        labels = batch.label - 1
        
        #loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        
        self.log('test_acc', acc) # By default logs it per epoch (weighted average over batches)