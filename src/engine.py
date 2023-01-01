import torch

from sklearn.metrics import accuracy_score

from tqdm import tqdm

import config

def train_one_batch(model, data, loss_fn, optimizer):
    optimizer.zero_grad()
    
    for k, v in data.items():
        data[k] = v.to(config.DEVICE)
    
    y = data['y'].unsqueeze(1)

    probs = model(data['X'], data['X_len'])
    
    with torch.no_grad():
        pred = probs.detach().cpu().sigmoid()
        pred = pred.numpy()>0.5
        acc = (pred==y.detach().cpu().numpy()).sum()/len(pred)

    loss = loss_fn(probs, y)
    
    loss.backward()
    optimizer.step()
    
    return loss, acc


def train_loop(model, data_loader, loss_fn, optimizer):
    model.train()
    total_loss = 0
    total_acc = 0
    num_batches = len(data_loader)
    loop = tqdm(data_loader)
    
    for data in loop:
        batch_loss, batch_score = train_one_batch(model, data, loss_fn, optimizer)
        
        with torch.no_grad():
            total_loss += batch_loss.item()
            total_acc += batch_score
        
        loop.set_postfix(dict(
                loss = batch_loss.item(),
                accuracy = batch_score
            ))
        
    avg_loss = round(total_loss/num_batches, 3)
    avg_acc = round(total_acc/num_batches, 3)

    return avg_loss, avg_acc

def validate_one_batch(model, data, loss_fn):
    for k, v in data.items():
        data[k] = v.to(config.DEVICE)
    
    y = data['y'].unsqueeze(1)
    
    probs = model(data['X'], data['X_len'])

    pred = probs.detach().cpu().sigmoid()
    pred = pred.numpy()>0.5
    acc = (pred==y.detach().cpu().numpy()).sum()/len(pred)
    
    loss = loss_fn(probs, y)
    
    return loss, acc

def validate_loop(model, data_loader, loss_fn):
    model.eval()

    num_batches = len(data_loader)
    loop = tqdm(data_loader)

    total_loss = 0
    total_score = 0
    
    with torch.no_grad():
        for data in loop:
                loss, acc = validate_one_batch(model, data, loss_fn)

                total_loss += loss.item()
                total_score += acc
                
                loop.set_postfix(dict(loss = loss.item(), accuracy=acc))
    
    val_acc = round(total_score/num_batches, 3)
    val_loss = round(total_loss/num_batches, 3)
    return val_loss, val_acc