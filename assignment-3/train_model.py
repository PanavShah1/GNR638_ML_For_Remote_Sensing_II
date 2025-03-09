import datetime
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path
import sys

def train_one_epoch(model, training_loader, optimizer, loss_fn, epoch_index, tb_writer, BATCH_SIZE, print_freq, output_loc, timestamp):
    running_loss = 0.
    last_loss = 0.
    running_acc = 0.
    last_acc = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        acc = (outputs.argmax(dim=1) == labels).sum().item() / BATCH_SIZE

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        running_acc += acc
        if i % (print_freq) == (print_freq - 1):
            last_loss = running_loss / (print_freq) # loss per batch
            last_acc = running_acc / (print_freq) # acc per batch
            print('  batch {} loss: {} acc {}'.format(i + 1, last_loss, last_acc))
            with open(output_loc / 'outputs/{}.txt'.format(timestamp), 'a') as f:
                f.write('  batch {} loss: {} acc {}\n'.format(i + 1, last_loss, last_acc))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            tb_writer.add_scalar('Accuracy/train', last_acc, tb_x)
            running_loss = 0.
            running_acc = 0.

    return last_loss, last_acc



def train(model, train_loader, val_loader, optimizer, loss_fn, batch_size, num_epochs, output_loc, resume=False, print_freq=10, timestamp=None):
    if resume:
        model.load_state_dict(torch.load(resume))
    # Initializing in a separate cell so we can easily add more epochs to the same run
    output_loc = Path(output_loc)
    os.makedirs(output_loc, exist_ok=True)
    os.makedirs(output_loc / 'models', exist_ok=True)
    os.makedirs(output_loc / 'runs', exist_ok=True)
    os.makedirs(output_loc / 'outputs', exist_ok=True)

    
    writer = SummaryWriter(output_loc / 'runs/fashion_trainer_{}'.format(timestamp))

    epoch_number = 0

    best_vloss = 1_000_000.

    for epoch in range(num_epochs):
        print('EPOCH {}:'.format(epoch_number + 1))
        with open(output_loc / 'outputs/{}.txt'.format(timestamp), 'a') as f:
            f.write('EPOCH {}:'.format(epoch_number + 1))            

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss, avg_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, epoch, writer, batch_size, print_freq, output_loc, timestamp)


        running_vloss = 0.0
        running_vacc = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss.item()
                vacc = (voutputs.argmax(dim=1) == vlabels).sum().item() / batch_size
                running_vacc += vacc

        avg_vloss = running_vloss / (i + 1)
        avg_vacc = running_vacc / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        print('ACC train {} valid {}'.format(avg_acc, avg_vacc))
        with open(output_loc / 'outputs/{}.txt'.format(timestamp), 'a') as f:
            f.write('LOSS train {} valid {}\n'.format(avg_loss, avg_vloss))
            f.write('ACC train {} valid {}\n'.format(avg_acc, avg_vacc))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.add_scalars('Training vs. Validation Accuracy',
                        { 'Training' : avg_acc, 'Validation' : avg_vacc },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = output_loc / 'models/model_{}_{}.pth'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1


    return model 

def test(model, test_loader, loss_fn, batch, output_loc, timestamp):
    output_loc = Path(output_loc)
    running_loss = 0.
    running_acc = 0.

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            acc = (outputs.argmax(dim=1) == labels).sum().item() / batch
            running_acc += acc
            avg_loss = running_loss / (i + 1)
            avg_acc = running_acc / (i + 1)

        print('TEST LOSS: {} ACC: {}'.format(avg_loss, avg_acc))
        with open(output_loc / 'outputs/{}.txt'.format(timestamp), 'a') as f:
            f.write('TEST LOSS: {} ACC: {}\n'.format(avg_loss, avg_acc))
