import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os
from pathlib import Path

from time import time
from datetime import datetime

from model import MLP


# Utility Methods
def print_epoch_result(result):
    print('Epoch Summary')
    print('Epoch: {}'.format(result[0]))
    print('Loss: {:.4f}'.format(result[1]))
    print('Number Correctly Classified: {}'.format(result[2]))
    print('Accuracy: {:.2f}%'.format(result[3]))
    print()


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Paths
PROJ_DIR = Path('.')
DATA_DIR = PROJ_DIR / 'DATA' / 'MNIST'
DATE = datetime.now().strftime('%Y-%b-%d@%H;%M;%S')
MODEL_DIR = PROJ_DIR / 'MODEL' / f'{DATE}'
MODEL_NAME = 'mlp_model.pt'

# Set this to True if the data has not been downloaded
download = False

# Model Parameters
# Based off Ciresan et al. Neural Computation 10, 2010 and arXiv 1003.0358, 2010
# http://yann.lecun.com/exdb/mnist/
in_channels = 784
hidden_layers = [2500, 2000, 1500, 1000, 500]
output = 10

# Training Parameters
lr = 0.003
momentum = 0.9
epochs = 10
batch_size = 64
shuffle = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Transform The Data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                                ])

print("----- LOADING MNIST DATA -----")

# Load the Data
make_dir(DATA_DIR)
train_data = datasets.MNIST(DATA_DIR, download=download, train=True, transform=transform)
test_data = datasets.MNIST(DATA_DIR, download=download, train=False, transform=transform)

# Prepare the Data Loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)

# Size of Data
training_size = len(train_loader) * train_loader.batch_size
testing_size = len(test_loader) * test_loader.batch_size
print("Training Size:", training_size)
print("Testing Size:", testing_size)
print()

print('----- MODEL -----')
model = MLP(in_channels=in_channels, hidden_layers=hidden_layers, output=output)
print(model)
print()

print('----- TRAINING PARAMETERS -----')
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
print('Optimizers and Criterion Setup')
print(f'Learning Rate: {lr}, Momentum Rate: {momentum}, Epochs: {epochs}, Shuffle: {shuffle}, Device: {device}')
print()

print('----- TRAINING -----')
# Summary of Training Results for Graphing
# [epoch, loss, number correctly classified, accuracy]
training_results = []

# Send Model and Loss Function to Device (CPU or GPU if available)
model.to(device)
criterion.to(device)

start_train_time = time()
for e in range(epochs):
    running_loss, train_corr = 0, 0
    for idx, (data, label) in enumerate(train_loader):
        idx += 1
        # Pre-process Data/ Flatten Image
        data = data.view(data.shape[0], -1)

        # Send Data and Label to Device (CPU or GPU if available)
        data = data.to(device)
        label = label.to(device)

        # Apply the Model
        output = model(data)
        loss = criterion(output, label)

        # Tally the number of correct predictions
        predicted = torch.max(output.data, 1)[1]
        batch_corr = (predicted == label).sum()
        train_corr += batch_corr

        # Update the Parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Add the Running Loss
        running_loss += loss.item()

        # Print Interim Results
        if idx % 100 == 0:
            print('Loss: {l_item:.4f}, Accuracy: {acc:.2f}%'.format(
                l_item=running_loss,
                acc=train_corr.item() * 100 / (100 * idx)
            ))
    else:
        accuracy = train_corr.item() * 100 / (100 * idx)
        print('Summary - Epoch: {epc}, Loss: {l_item:.4f}, Accuracy: {acc:.2f}%'.format(
            epc=e,
            l_item=running_loss,
            acc=accuracy
        ))
        print()
        training_results.append([e, running_loss, train_corr.item(), accuracy])
train_duration = time() - start_train_time
print('Training Duration: {:.2f}s'.format(train_duration))
print('Final Training Accuracy: {:.2f}%'.format(training_results[len(training_results) - 1][3]))
print()

print('----- TESTING -----')
test_corr = 0
start_test_time = time()
with torch.no_grad():
    for _, (data, label) in enumerate(test_loader):
        # Pre-process Data/ Flatten Image
        data = data.view(data.shape[0], -1)

        # Send Data to Device
        data = data.to(device)
        label = label.to(device)

        # Apply model
        output = model(data)
        predicted = torch.max(output.data, 1)[1]
        batch_corr = (predicted == label).sum()
        test_corr += batch_corr

test_duration = time() - start_test_time
print('Testing Duration:', test_duration, 's')
print("Accuracy: {:.2%}".format(test_corr.item() / testing_size))
print()

print('----- SAVING MODEL -----')
make_dir(MODEL_DIR)
torch.save(model.state_dict(), MODEL_DIR / MODEL_NAME)
print('Path:', MODEL_DIR / MODEL_NAME)
print("DONE")
