import spacy

from models import AvgWordEmbeddings, LstmEncoders

spacy_eng = spacy.load('en_core_web_sm')
model_dict = {}
model_dict["word_embs"] = AvgWordEmbeddings
model_dict["uniLSTM"] = LstmEncoders
model_dict["biLSTM"] = LstmEncoders
model_dict["biLSTMPool"] = LstmEncoders

def create_model(model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert False, "Unknown model name \"%s\". Available models are: %s" % (model_name, str(model_dict.keys()))      
        
def tokenizer(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]