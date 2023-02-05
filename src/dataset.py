import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator

from sklearn.model_selection import train_test_split

from utils import load_vocab

import config

class Dataset:
    def __init__(self, data, label_pipeline, text_pipeline):
        self.data = data
        self.label_pipeline = label_pipeline
        self.text_pipeline = text_pipeline
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]
        
        label = self.label_pipeline(label)
        indxd_text = self.text_pipeline(text)
        
        return {
            'X': torch.tensor(indxd_text, dtype=torch.long),
            'y': torch.tensor(label, dtype=torch.float32)
        }

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    def __call__(self, batch):
        labels = torch.tensor([item['y'].item() for item in batch])
        indxd_tokens = [item['X'] for item in batch]
        
        data_len = torch.tensor([len(item['X']) for item in batch])
        padded_tokens = pad_sequence(indxd_tokens, batch_first=True,
                                    padding_value=self.pad_idx)
       
        return {'X': padded_tokens, 'y': labels, 'X_len':data_len}

def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield text

def get_loaders(data, load=False):
    train_set, val_set = train_test_split(data, test_size=0.25, shuffle=True,
                                        stratify=data['sentiment'], random_state=42)

    train_set = train_set.values
    val_set = val_set.values

    if load==False:
        train_iter = iter(train_set)

        vocab = build_vocab_from_iterator(yield_tokens(train_iter), min_freq=config.MIN_COUNT,
                                    specials=["<unk>", "<pad>"])

        #This index will be returned when OOV token is queried.
        vocab.set_default_index(vocab["<unk>"])
    else:
        vocab = load_vocab()

    text_pipeline = lambda x: vocab(x)
    label_pipeline = lambda x: int(x)

    train_data = Dataset(train_set, label_pipeline, text_pipeline)
    val_data = Dataset(val_set, label_pipeline, text_pipeline)


    train_loader = torch.utils.data.DataLoader(train_data, batch_size = config.BATCH_SIZE, 
                                            pin_memory=config.PIN_MEMORY, shuffle=True,
                                            collate_fn=MyCollate(vocab.lookup_indices(["<pad>"])[0]))

    val_loader = torch.utils.data.DataLoader(val_data, batch_size = config.BATCH_SIZE,
                                        pin_memory=config.PIN_MEMORY, shuffle=False,
                                        collate_fn=MyCollate(vocab.lookup_indices(["<pad>"])[0]))

    return train_loader, val_loader, vocab