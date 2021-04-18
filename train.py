import os
import argparse
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint,EarlyStopping

from task import InferSent
from dataset import create_iterators


CHECKPOINT_PATH = "../saved_models"

model_hparams = {"word_embs": {"freeze": True}, # can be modified later
                "uniLSTM": {"freeze": True,"max_pool":False, "bidirectional":False,
                     "lstm_input_size":300, "lstm_hidden_size":2048, "num_layers" :1},
                "biLSTM": {"freeze": True,"max_pool":False, "bidirectional":True,
                     "lstm_input_size":300, "lstm_hidden_size":2048, "num_layers" :1},
                "biLSTMPool": {"freeze": True,"max_pool":True, "bidirectional":True,
                     "lstm_input_size":300, "lstm_hidden_size":2048, "num_layers" :1}
                    }

optimizer_hparams = {"lr": 0.1}                    


def train_model(args):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    if args.save_name=="":
        save_name = args.model_name

    TEXT, train_iter, val_iter, test_iter = create_iterators(batch_size = args.batch_size)

    # Create a PyTorch Lightning trainer with the generation callback
    if args.resume_path == "":
      trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),                                  # Where to save models
                          checkpoint_callback = ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"), # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                          gpus=1 if "cuda" in str(args.device) else 0,                                                     # We run on a single GPU (if possible)
                          max_epochs = args.max_epochs,                                                                            # How many epochs to train for if no patience is set
                          callbacks=[LearningRateMonitor("epoch"), 
                                      EarlyStopping(monitor='val_acc',patience=3,mode='max')],                                                   # Log learning rate every epoch
                          progress_bar_refresh_rate=0)   # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
    else:
      trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),                                  # Where to save models
                          checkpoint_callback = ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"), # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                          gpus=1 if "cuda" in str(args.device) else 0,                                                     # We run on a single GPU (if possible)
                          max_epochs = args.max_epochs,                                                                            # How many epochs to train for if no patience is set
                          callbacks=[LearningRateMonitor("epoch"), 
                                      EarlyStopping(monitor='val_acc',patience=3,mode='max')],                                                   # Log learning rate every epoch
                          progress_bar_refresh_rate=0,
                          resume_from_checkpoint = args.resume_path)   # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
    
    
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
    
    if args.resume_path != "": #os.path.isfile(pretrained_filename):
        pl.seed_everything(42) # To be reproducable
        print("Found pretrained model at %s, loading..." % args.resume_path)
        model = InferSent.load_from_checkpoint(args.resume_path) # Automatically loads the model with the saved hyperparameters
        trainer.fit(model, train_iter, val_iter)
   
    else:
        
        pl.seed_everything(42) # To be reproducable

        model_params = model_hparams[args.model_name]
        #print("model params", model_params)
        model_params["embeddings"] = TEXT.vocab.vectors
        if args.model_name != "word_embs": # enable setting LSTM params
            model_params["lstm_hidden_size"] = args.lstm_hidden_size
            model_params["num_layers"] = args.num_layers
            model_params["freeze"] = args.freeze

        model = InferSent(model_name=args.model_name, model_hparams=model_params,
                                optimizer_name = args.optimizer_name, optimizer_hparams = 
                                {"lr": args.lr})

        trainer.fit(model, train_iter, val_iter)
        model = InferSent.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, test_dataloaders=val_iter, verbose=False)
    test_result = trainer.test(model, test_dataloaders=test_iter, verbose=False)
    result = {"test": test_result, "val": val_result}

    print(f"The best model on validation set achieves test acc {test_result['test_acc']} and val_acc {test_result['test_acc']}")
    
    return model, result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument("--model_name", default = 'word_embs', choices = ["word_embs", "uniLSTM",
        "biLSTM", "biLSTMPool"], help = "What kind of model to use.")

    parser.add_argument("--lstm_hidden_size", type = int, default=2048,
                        help = "Dimensionality of the LSTM hidden layer")
    parser.add_argument("--num_layers", type = int, default=1,
                        help = "Number of LSTM layers to apply")
    parser.add_argument("--freeze", type = bool, default=True,
                        help = "If True, the embedding layer should be non-trainable.")
                    
    
    # Training params
    parser.add_argument('--optimizer_name', type=str, default="SGD", choices = ["Adam", "SGD"],
                        help='Optimizer to use')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of examples to process in a batch')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')  
    parser.add_argument("--weight_decay", type = float, default = 0.99,
                         help = "Learning rate decay.")                       
    parser.add_argument('--max_epochs', type=float, default=25,
                        help='Max number of epochs to train for')  

    # Misc params
    parser.add_argument('--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
                        help="Device to run the model on.")
    parser.add_argument('--save_name', type=str, default="",
                        help="The name of the saved model. If empty string, defaults to model_name")

    parser.add_argument('--resume_path', type=str, default="",
                        help="Path to resume training.")

    args = parser.parse_args()

    print(f"[INFO]: Using device {args.device}", flush=True)
    print("[INFO]: Using following parameeters:", args)
    train_model(args)

    