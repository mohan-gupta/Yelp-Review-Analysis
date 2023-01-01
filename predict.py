import torch
import torch.nn as nn

from src import config
from src.utils import load_model, load_vocab
from src.preprocess import tokenizer

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
        
    def forward(self, x):
        x_embed = self.embedding(x)
        
        output, hn = self.gru(x_embed)
        
        hn = hn[-1]
        
        x = self.dropout(hn)
        x = self.out(x)
        
        return x


vocab = load_vocab()

text_pipeline = lambda txt: vocab(txt)

model = Model(len(vocab), config.INPUT_SIZE, config.HIDDEN_SIZE,
                 config.OUTPUT_SIZE, config.DROP, config.NUM_LAYERS)

model = load_model(model, map_location="cpu")


def get_prediction(text):
    _, tokens = tokenizer((-1, text))

    if len(tokens)==0:
        return (-1, "Enter Proper Text")

    inputs = torch.tensor(text_pipeline(tokens), dtype=torch.long)

    logits = model(inputs)

    return (1, torch.sigmoid(logits).item())