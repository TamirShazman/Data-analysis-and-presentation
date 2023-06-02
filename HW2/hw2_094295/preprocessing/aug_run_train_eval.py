import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from tqdm import tqdm

def imshow(inp, title=None):
    """Imshow for Tensors."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(15, 15))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=100):
    """Responsible for running the training and validation phases for the requested model."""
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    loss_dict = {'train': [], 'val': []}
    acc_dict = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            acc_dict[phase].append(epoch_acc.item())
            loss_dict[phase].append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # If the current epoch provides the best validation accuracy so far, save the model's weights.
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, loss_dict, acc_dict

def test_model(model, test_dl, device, criterion, dataset_size):
    running_loss = 0.0
    running_corrects = 0

    model.eval() 

    for inputs, labels in tqdm(test_dl):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # track history only in train
                with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size

    return epoch_loss, epoch_acc

def train_fold(train_dl, val_dl, test_dl, num_epochs, lr, num_of_classes, exp_path):
    np.random.seed(0)
    torch.manual_seed(0)

    print("Your working directory is: ", os.getcwd())




    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloaders = {'train': train_dl, 'val': val_dl}
    dataset_sizes = {'train': len(train_dl.dataset), 'val': len(val_dl.dataset)}


    # Use a prebuilt pytorch's ResNet50 model
    model_ft = models.resnet50(pretrained=False)

    # Fit the last layer for our specific task
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_of_classes)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.Adam(model_ft.parameters(), lr=lr)

    # Decay LR by a factor of 0.1 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

    # Train the model
    model_ft, loss_dict, acc_dict = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, device, num_epochs)

    # Save the trained model
    models_dir = os.path.join(exp_path, 'models')
    os.makedirs(models_dir, exist_ok=True)
    torch.save(model_ft.state_dict(), os.path.join(models_dir, "trained_model.pt"))

    figures_dir = os.path.join(exp_path, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Basic visualizations of the model performance
    fig = plt.figure(figsize=(20,10))
    plt.title("Train - Validation Loss")
    plt.plot(loss_dict['train'], label='train')
    plt.plot(loss_dict['val'], label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc='best')
    plt.savefig(os.path.join(figures_dir, 'train_val_loss_plot.png'))

    fig = plt.figure(figsize=(20,10))
    plt.title("Train - Validation ACC")
    plt.plot(acc_dict['train'], label='train')
    plt.plot(acc_dict['val'], label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('ACC', fontsize=12)
    plt.legend(loc='best')
    plt.savefig(os.path.join(figures_dir, 'train_val_acc_plot.png'))

    test_loss, test_acc = test_model(model_ft, test_dl, device, criterion, len(test_dl.dataset))

    with open(exp_path + '/fold_result.txt', 'w') as f:
        f.write(f'loss: {test_loss}, Acc: {test_acc}')

    return test_loss, test_acc


