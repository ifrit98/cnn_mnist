#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 22:05:36 2018

@author: jason st george
"""
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import matplotlib.ticker as ticker
import argparse

#####################################################################################################

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 320) # self.num_flat_features(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

#####################################################################################################

def train(args, net, device, trainloader, optimizer, criterion, epoch):   
    epoch_loss = 0.
    net.train()

    dataset_sz = len(trainloader.dataset)
    loader_sz = len(trainloader)
    
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
 
        epoch_loss += loss.item()
        if i % args.log_interval == 0:
            print('Training Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(data), dataset_sz,
                    100. * i / loader_sz, loss.item()))
                
    return epoch_loss

#####################################################################################################

def test(epochs, net, device, testloader, criterion):
    net.eval()
    
    correct = 0
    test_loss = 0
    total = len(testloader.dataset)
    
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    
    test_loss /= total
    print('\nTest: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, total, 100. * correct / total))
    
    return test_loss

#####################################################################################################

def plot(data, saved_losses, saved_test_losses, spreads, class_acc, epochs, ofit_epoch):
    # Loss plot
    fig, ax = plt.subplots()
    
    x = np.linspace(1, epochs, epochs)
    saved_losses = np.array(saved_losses)
    test_losses = np.array(saved_test_losses)
    test_losses *= 4500 # Ratio to normalize test to line up with train loss (900/.2)
    
    ax.set_title("Average Model Loss over " + str(epochs) + " Epochs")

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Average Loss")
    
    tick_spacing = 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    
    ax.plot([ofit_epoch, ofit_epoch], [min(test_losses), max(test_losses)])
    ax.plot(x, saved_losses, label='Train', color='purple', marker=".")
    ax.plot(x, test_losses, label='Test', color='green', marker="x")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    with open('model_loss.pkl','wb') as fid:
        pickle.dump(ax, fid)
    fig.savefig('./MNISTmodel_loss')

    arr = []
    for i in range(10):
        arr.append(class_acc[i][0] / class_acc[i][1])

    x = np.arange(10)
    
    # Histogram plot of all 10 classes
    hfig, az = plt.subplots()
    plt.title('Relative accuracy of all classes')
    plt.bar(x, height=arr)
    plt.xticks(x);
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    hfig.savefig( 'MNIST_accuracy_hist.pdf', bbox_inches='tight')
    plt.show()

    # Plots of each class predictions
    for i in range(10):    
        hfig, az = plt.subplots()
        plt.title('Prediction spread of class: ' + str(i))
        plt.bar(x, height=spreads[str(i)])
        plt.xticks(x);
        plt.xlabel('Classes')
        plt.ylabel('# Predicted')
        hfig.savefig( 'MNIST_class_' + str(i) + '_accuracy_hist.pdf', bbox_inches='tight')
        plt.show()
       

#####################################################################################################    

def class_accuracy(testloader, device, net, test_size):
    test_correct = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, 
                    '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    test_total = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, 
                    '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
   
    spreads = {'0': [0 for i in range(10)], '1': [0 for i in range(10)], 
               '2': [0 for i in range(10)], '3': [0 for i in range(10)], 
               '4': [0 for i in range(10)], '5': [0 for i in range(10)], 
               '6': [0 for i in range(10)], '7': [0 for i in range(10)], 
               '8': [0 for i in range(10)], '9': [0 for i in range(10)]}

    c_samples, i_samples = 0, 0
    cor_done = {'0': False, '1': False, '2': False, '3': False, '4': False, 
                    '5': False, '6': False, '7': False, '8': False, '9': False}
    incor_done = {'0': False, '1': False, '2': False, '3': False, '4': False, 
                    '5': False, '6': False, '7': False, '8': False, '9': False}

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).squeeze()
            np_predicted = predicted.cpu().numpy()
            for i in range(len(outputs)):            
                label = str(labels[i])
                label = label[7]
                test_correct[label] += correct[i].item()
                test_total[label] += 1
                
                if np_predicted[i] != int(label):
                    spreads[label][np_predicted[i]] += 1
                    if i_samples < 10:
                        i_samples += save_samples(images[i], np_predicted[i], label,
                                                  i_samples, incor_done, False)
                else:
                    if c_samples < 10:
                        c_samples  += save_samples(images[i], None, label, 
                                                   c_samples, cor_done, True)
                                                
        for i in range(10):
            spreads[str(i)][i] = test_correct[str(i)]
                
    accuracies = []
    for i in range(10):
        accuracies.append((test_correct[str(i)], test_total[str(i)]))
        print('Accuracy of %5s : %2d %%' % (
            i, 100 * test_correct[str(i)] / test_total[str(i)]))

    return spreads, accuracies

#####################################################################################################    
    
def save_samples(tensor, predicted, label, samples, done, correct):
    if done[label]:
        return 0
    done[label] = True

    inp = tensor.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()
    
    if correct:
        plt.title('Correctly labeled: ' + str(label))
        plt.imsave('correct_' + str(label), inp)

    else:
        plt.title('Incorrectly labeled: ' + str(predicted))
        plt.imsave('incorrect_' + str(label), inp)

    return 1

#####################################################################################################    

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

#####################################################################################################    
    
def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=25, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test', action='store_true', default=False,
                        help='disables training, loads best model')

    args = parser.parse_args()
        
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)

    torch.manual_seed(args.seed)
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    train_ds = datasets.MNIST('../data', train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ]))
    
    test_ds = datasets.MNIST('../data', train=False, download=True,
           transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
           ]))

    trainloader = torch.utils.data.DataLoader(train_ds,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    testloader = torch.utils.data.DataLoader(test_ds,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    test_size = len(test_ds)
    
    if args.test:
        net = torch.load('MNISTmodel@93')
    else:
        net = Net()
    net.to(device)
        
    # Equivalent to cross entropy loss with softmax
    criterion = F.nll_loss
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    
    saved_losses, saved_test_losses = [], []
    last_test_loss, snd_last_test_loss = 0, 0
    overfit_count = 0
    
    for epoch in range(1, args.epochs+1):
        if not args.test:
            train_loss = train(args, net, device, trainloader, optimizer, criterion, epoch)
            saved_losses.append(train_loss)
        
        test_loss = test(args.epochs, net, device, testloader, criterion)
        saved_test_losses.append(test_loss)
   
        if test_loss > last_test_loss and last_test_loss > snd_last_test_loss:
            overfit_count += 1
            if overfit_count == 4:
                print('Overfitting starting at: ' + str(epoch) + ' epochs')
                overfit_epoch = epoch
                torch.save(net, './MNISTmodel@' + str(epoch))

        snd_last_test_loss = last_test_loss
        last_test_loss = test_loss
        
    torch.save(net, './MNISTmodelFullyTrained@' + str(args.epochs) + ' epochs')
    
    class_spreads, class_acc = class_accuracy(testloader, device, net, test_size)
    plot(testloader, saved_losses, saved_test_losses, class_spreads, class_acc, 
         args.epochs, overfit_epoch)
    
    
if __name__ == "__main__":
    main()
   
