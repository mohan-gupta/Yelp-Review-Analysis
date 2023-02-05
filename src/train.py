import torch
import torch.nn as nn

from model import Model

import pickle

import config
from utils import get_processed_data, save_model, load_model
from dataset import get_loaders
from engine import train_loop, validate_loop

def main(save=False, load=False):
    start = 0

    data = get_processed_data()

    train_loader, val_loader, vocab = get_loaders(data, load=load)

    net = Model(len(vocab), config.INPUT_SIZE, config.HIDDEN_SIZE,
                 config.OUTPUT_SIZE, config.DROP, config.NUM_LAYERS)

    net.to(config.DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(net.parameters(), lr=config.LR,
                                    weight_decay=config.DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                            threshold=1e-2, patience=3,
                                                            verbose=True)
    if load:
        start, net, optimizer, scheduler = load_model(net, optimizer)

    for epoch in range(start, start+config.EPOCHS):
        print(f"Epoch:{epoch+1}")
        train_loss, train_acc = train_loop(net, train_loader, loss_fn, optimizer)
        val_loss, val_acc = validate_loop(net, val_loader, loss_fn)
        scheduler.step(val_loss)

        print(f"Training Loss={train_loss}, Accuracy={train_acc},\n\
            Validation Loss = {val_loss} Accuracy={val_acc}")

    if save:
        save_model(epoch, net, optimizer, scheduler)

        with open(config.VOCAB_PATH, 'wb') as f:
            vocab_data = {"vocab": vocab}
            pickle.dump(vocab_data, f)

if __name__ == "__main__":
    main(save=True, load=False)