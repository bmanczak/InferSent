## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

class AvgWordEmbeddings(nn.Module):

    def __init__(self, embeddings, freeze = True):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=freeze)

    def forward(self, x):
        emb = self.embedding(x) # [seq_len, batch_size, input_size]
        return torch.mean(emb, axis = 0) # mean over the seq_len

class LstmEncoders(nn.Module):
    """
    Models LSTM based encoders. 

    Inputs:
    ----------

    TEXT: torchtext.data.Field
        Object with the the 'build_vocab' method already built
    lstm_input_size: int
        Dim of the embeddings.
    lstm_hidden_size: int
        Hidden size of the lstm.
    num_layers: int
        Number of layers
    bidirectional: bool
        If True, becomes a bidirectional LSTM
    freeze: bool
        If True, the embedding layer is trainable.
    """

    def __init__(self, embeddings, max_pool = False, lstm_input_size=300, lstm_hidden_size=2048, num_layers = 1,
                    bidirectional = False, freeze = True):

        super().__init__()
        self.max_pool = max_pool
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        if freeze==True:
          print("[INFO]: Freezing the embeddings")
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=freeze)
        self.lstm =  nn.LSTM(lstm_input_size, lstm_hidden_size, num_layers, bidirectional = bidirectional)
    
    def forward(self, x, input_lengths):
        """
        Runs the forward method and concatenates the last hidden state in various ways.

        Inputs:

        x: Tensor
            Input tensor of size [seq_len, batch_size]
        input_lenght: Tensor
            Tensor holding unpadded input lentghs.
        concat_type: str
            Defines the final operation on the hidden states.
            If a bidirectional LSTM is used, the hidden states are concatenated.
            Supported: ["last", "max_pool"]
        """

        batch_size = x.shape[1]

        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, input_lengths.cpu(), enforce_sorted=False)
        outputs, hidden = self.lstm(packed) # h = (hidden_state, cell_state) for t=seq_len

        hidden_state = hidden[0].view(self.num_layers, self.num_directions, batch_size, self.lstm_hidden_size)

        if self.num_directions==1:
            return hidden_state[-1, 0, :, :] # [batch_size, hidden_dim]
        
        elif (self.num_directions==2) and (not self.max_pool): # bidirectional
            return torch.cat((hidden_state[-1, 0, :, :], hidden_state[-1, 1, :, :]), axis = 1)
        
        elif (self.num_directions==2) and (self.max_pool):
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
            outputs = outputs.view(-1, batch_size, 2, self.lstm_hidden_size)
            outputs = torch.cat((outputs[:,:,0,:], outputs[:,:,1,:]), axis = -1) # [seq_len, batch_size, 2*hidden_size]
            return torch.max(outputs, dim = 0)[0]

        else: 
            raise ValueError("Forward type not supported! Please use 'last' or 'max_pool'")

"""
Resources:
Packing thee sequence: https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec , https://discuss.pytorch.org/t/packedsequence-for-seq2seq-model/3907
LSTM doc: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
torchtext doc: https://torchtext.readthedocs.io/en/latest/data.html?highlight=build_vocab#torchtext.data.Field.build_vocab
"""

        