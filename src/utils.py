import pandas as pd
import pickle
import torch
import src.config as config
import os

CWD = os.path.dirname(__file__)

def get_data():
    df = pd.read_csv("../dataset/data.csv")

    return df

def get_processed_data():
    df = pd.read_csv('../dataset/processed_data.csv')

    df['text'] = df['text'].str.split()
    
    return df

def save_model(epoch, model, optimizer, scheduler):
    checkpoint = {"epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler}
    torch.save(checkpoint, config.CHECKPOINT_PATH)

def load_model(model, optimizer=None, map_location="cuda"):
    loc = torch.device("cuda")
    if map_location == "cpu":
        loc = torch.device("cpu")
    
    path = os.path.join(CWD, config.CHECKPOINT_PATH)

    checkpoint = torch.load(path, map_location = loc)

    model.load_state_dict(checkpoint['model'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch'], model, optimizer, checkpoint['scheduler']

    return model

def load_vocab():
    path = os.path.join(CWD, config.VOCAB_PATH)
    
    with open(path , 'rb') as f:
        vocab = pickle.load(f)
    
    return vocab['vocab']