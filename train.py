import argparse
import time

import torch
from torch import optim, nn
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

from senn import SENN
from reg import parametriser_regulariser

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    start_time = time.time()
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        data.requires_grad = True
        g, h, theta = model(data)
        cls_loss = criterion(g, label)
        reg = parametriser_regulariser(data, g, theta, h)
        total_loss = cls_loss + 2e-4 * reg
        total_loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                       batch_idx * len(data),
                                                                       len(train_loader.dataset),
                                                                       100. * batch_idx / len(train_loader),
                                                                       total_loss.item()))
    print('Train Epoch: {}\tTime {}'.format(epoch, time.time() - start_time))

def val(model, device, valloader, criterion):
    model.eval()
    val_loss = 0
    correct = 0

    for data, label in valloader:
        data, label = data.to(device), label.to(device)
        data.requires_grad = True
        g, h, theta = model(data)
        cls_loss = criterion(g, label)
        reg = parametriser_regulariser(data, g, theta, h)
        total_loss = cls_loss + 2e-4 * reg
        val_loss += total_loss.sum().item()  # sum up batch loss
        pred = g.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(label.view_as(pred)).sum().item()

    val_loss /= len(valloader.dataset)
    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(val_loss,
                                                                                correct,
                                                                                len(valloader.dataset),
                                                                                100. * correct / len(valloader.dataset)))
    return val_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch SENN')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate for Adam optimiser')
    parser.add_argument('--seed', type=int, default=5, metavar='S', help='random seed (default: 1)')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    valset = datasets.MNIST('../data', train=False, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, drop_last=True, **kwargs)

    model = SENN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.NLLLoss()

    best_val_loss = 0.0
    for epoch in range(1, args.epochs + 1):
        train(model, device, trainloader, optimizer, criterion, epoch)
        val_loss = val(model, device, valloader, criterion)

        if val_loss > best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "senn_mnist_best_model.pt")


if __name__ == '__main__':
    main()
