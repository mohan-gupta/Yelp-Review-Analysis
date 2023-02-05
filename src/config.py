import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MIN_COUNT = 5

BATCH_SIZE = 1024
PIN_MEMORY = True
INPUT_SIZE = 100
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1
NUM_LAYERS = 1
DROP = 0.1
LR = 1e-3
DECAY = 0.01
EPOCHS = 20
CHECKPOINT_PATH = "../model/checkpoint.pt"
VOCAB_PATH = "../model/vocab.pkl"