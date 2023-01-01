import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class Model(nn.Module):
    def __init__(self, vocab_size, input_size,
                hidden_size, output_size, dropout=0.1, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, input_size)

        self.gru = nn.GRU(input_size=input_size, hidden_size = hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)

        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, x_len):
        x_embed = self.embedding(x)
        
        #for ignoring the calculation of padding
        packed_input = pack_padded_sequence(x_embed, x_len.cpu().numpy(),
                                            batch_first=True, enforce_sorted=False)
        
        packed_output, hn = self.gru(packed_input.float())
        
        #to get the last layer hidden state in multilayered rnn
        #(num layers, batch size, hidden size)
        hn = hn[-1,:,:]
        
        x = self.dropout(hn)
        x = self.out(x)
        
        return x