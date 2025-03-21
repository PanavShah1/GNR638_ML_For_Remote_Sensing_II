import datetime
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path
import sys

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def train_one_epoch(model, training_loader, optimizer, loss_fn, epoch_index, tb_writer, BATCH_SIZE, print_freq, output_loc, timestamp):
    running_loss = 0.
    running_acc = 0.

    model.train()
    for i, data in enumerate(training_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        acc = (outputs.argmax(dim=1) == labels).sum().item() / BATCH_SIZE
        running_loss += loss.item()
        running_acc += acc

        if i % print_freq == (print_freq - 1):
            last_loss = running_loss / print_freq
            last_acc = running_acc / print_freq
            print(f'  batch {i + 1} loss: {last_loss} acc {last_acc}')
            with open(output_loc / f'outputs/{timestamp}.txt', 'a') as f:
                f.write(f'  batch {i + 1} loss: {last_loss} acc {last_acc}\n')
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            tb_writer.add_scalar('Accuracy/train', last_acc, tb_x)
            running_loss = 0.
            running_acc = 0.

    return last_loss, last_acc

def train(model, train_loader, val_loader, optimizer, loss_fn, batch_size, num_epochs, output_loc, resume=False, print_freq=10, timestamp=None):
    if resume:
        model.load_state_dict(torch.load(resume))
    
    output_loc = Path(output_loc)
    os.makedirs(output_loc / 'models', exist_ok=True)
    os.makedirs(output_loc / 'runs', exist_ok=True)
    os.makedirs(output_loc / 'outputs', exist_ok=True)

    writer = SummaryWriter(output_loc / f'runs/fashion_trainer_{timestamp}')
    model.to(device)
    epoch_number = 0
    best_vloss = float('inf')

    model_number = None

    for epoch in range(num_epochs):
        print(f'EPOCH {epoch_number + 1}:')
        with open(output_loc / f'outputs/{timestamp}.txt', 'a') as f:
            f.write(f'EPOCH {epoch_number + 1}:\n')
        
        avg_loss, avg_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, epoch, writer, batch_size, print_freq, output_loc, timestamp)
        
        running_vloss, running_vacc = 0.0, 0.0
        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss.item()
                vacc = (voutputs.argmax(dim=1) == vlabels).sum().item() / batch_size
                running_vacc += vacc

        avg_vloss = running_vloss / (i + 1)
        avg_vacc = running_vacc / (i + 1)
        print(f'LOSS train {avg_loss} valid {avg_vloss}')
        print(f'ACC train {avg_acc} valid {avg_vacc}')
        with open(output_loc / f'outputs/{timestamp}.txt', 'a') as f:
            f.write(f'LOSS train {avg_loss} valid {avg_vloss}\n')
            f.write(f'ACC train {avg_acc} valid {avg_vacc}\n')
        
        writer.add_scalars('Training vs. Validation Loss', {'Training': avg_loss, 'Validation': avg_vloss}, epoch_number + 1)
        writer.add_scalars('Training vs. Validation Accuracy', {'Training': avg_acc, 'Validation': avg_vacc}, epoch_number + 1)
        writer.flush()
        
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = output_loc / f'models/model_{timestamp}_{epoch_number}.pth'
            torch.save(model.state_dict(), model_path)
            if model_number is not None:
                os.remove(output_loc / f'models/model_{timestamp}_{model_number}.pth')
            model_number = epoch_number

        
        epoch_number += 1
    
    return model 

def test(model, test_loader, loss_fn, batch_size, output_loc, timestamp):
    output_loc = Path(output_loc)
    running_loss = 0.
    running_acc = 0.

    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            acc = (outputs.argmax(dim=1) == labels).sum().item() / batch_size
            running_acc += acc
            avg_loss = running_loss / (i + 1)
            avg_acc = running_acc / (i + 1)

        print(f'TEST LOSS: {avg_loss} ACC: {avg_acc}')
        with open(output_loc / f'outputs/{timestamp}.txt', 'a') as f:
            f.write(f'TEST LOSS: {avg_loss} ACC: {avg_acc}\n')
