import argparse

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

from senn import SENN


def val(model, device, valloader):
    model.eval()
    correct = 0

    for data, label in valloader:
        data, label = data.to(device), label.to(device)
        with torch.no_grad():
            g, h, theta = model(data)
        pred = g.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()

    print('\nVal set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct,
                                                          len(valloader.dataset),
                                                          100. * correct / len(valloader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch SENN')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--model-dict', type=str, default='senn_mnist_best_model.pt', help='pretrained model')
    parser.add_argument('--seed', type=int, default=5, metavar='S', help='random seed (default: 1)')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    valset = datasets.MNIST('../data', train=False, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, drop_last=True, **kwargs)

    model = SENN().to(device)
    model.load_state_dict(torch.load(args.model_dict))

    val(model, device, valloader)


if __name__ == '__main__':
    main()