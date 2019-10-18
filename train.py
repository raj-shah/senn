import argparse
import time
import numpy as np

import torch
from torch import optim, nn
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from senn import SENN
from reg import parametriser_regulariser

def train(writer, model, device, trainloader, robustness_optimizer, cls_optimizer, criterion, epoch):

    # epoch metrics
    correct = 0
    train_loss = 0.0

    model.train()
    start_time = time.time()

    for batch_idx, (data, label) in enumerate(trainloader):
        data, label = data.to(device), label.to(device)
        data.requires_grad = True

        # reset grad
        robustness_optimizer.zero_grad()
        cls_optimizer.zero_grad()

        # senn output
        g, h, theta = model(data)

        # loss + reg
        cls_loss = criterion(g, label)
        reg = parametriser_regulariser(data, g, theta, h)
        total_loss = cls_loss + 2e-4 * reg

        # update grad
        total_loss.backward()
        cls_optimizer.step()
        robustness_optimizer.step()

        # iteration metrics
        writer.add_scalar('loss/train', total_loss.item(), batch_idx)

        # update epoch metrics
        train_loss += total_loss.sum().item()
        pred = g.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(label.view_as(pred)).sum().item()

    train_loss /= len(trainloader.dataset)

    writer.add_scalar('accuracy/train', 100. * correct / len(trainloader.dataset), epoch)
    print('Epoch {}\t Train\t Loss: {:.4f} Accuracy: {}/{} ({:.4f}%)\t Time: {}'.format(epoch,
                                                                                        train_loss,
                                                                                        correct,
                                                                                        len(trainloader.dataset),
                                                                                        100. * correct / len(trainloader.dataset),
                                                                                        time.time() - start_time))

def val(writer, model, device, valloader, criterion, epoch):
    print('Epoch {}: Val'.format(epoch))

    # metrics
    val_loss = 0.0
    correct = 0

    model.eval()
    start_time = time.time()

    for batch_idx, (data, label) in enumerate(valloader):
        data, label = data.to(device), label.to(device)
        data.requires_grad = True
        g, h, theta = model(data)
        cls_loss = criterion(g, label)
        reg = parametriser_regulariser(data, g, theta, h)
        total_loss = cls_loss + 2e-4 * reg

        # iteration metrics
        writer.add_scalar('loss/val', total_loss.item(), batch_idx)

        # epoch metrics
        val_loss += total_loss.sum().item()  # sum up batch loss
        pred = g.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(label.view_as(pred)).sum().item()

    val_loss /= len(valloader.dataset)

    writer.add_scalar('accuracy/val', 100. * correct / len(valloader.dataset), epoch)
    print('Epoch {}\t Val\t Average loss: {:.4f}\t Accuracy: {}/{} ({:.0f}%)\t Time: {}'.format(epoch,
                                                                                                val_loss,
                                                                                                correct,
                                                                                                len(valloader.dataset),
                                                                                                100. * correct / len(valloader.dataset),
                                                                                                time.time() - start_time))
    return val_loss


def main():
    parser = argparse.ArgumentParser(description='PyTorch SENN')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate for Adam optimiser')
    parser.add_argument('--seed', type=int, default=1337, metavar='S', help='random seed (default: 1)')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    writer = SummaryWriter()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    valset = datasets.MNIST('../data', train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers,
                                            pin_memory=True)

    model = SENN().to(device)

    robustness_parameters = list(model.relevance_parametrizer.parameters())
    robustness_optimizer = optim.Adam(robustness_parameters, lr=args.lr)

    cls_paramteres = list(model.concept_encoder.parameters()) + list(model.aggregator.parameters())
    cls_optimizer = optim.Adam(cls_paramteres, lr=args.lr)

    criterion = nn.CrossEntropyLoss()

    best_val_loss = 0.0
    for epoch in range(1, args.epochs + 1):
        train(writer, model, device, trainloader, robustness_optimizer, cls_optimizer, criterion, epoch)
        val_loss = val(writer, model, device, valloader, criterion, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "senn_mnist_best_model.pt")


if __name__ == '__main__':
    main()