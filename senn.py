import torch
import torch.nn as nn
import torch.nn.functional as F

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class ConceptEncoder(nn.Module):
    def __init__(self):
        super(ConceptEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(20*20*20, 16)
        self.fc2 = nn.Linear(16,10)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = out.view(-1, num_flat_features(out))
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


## RelevanceParametrizer == LeNet
class RelevanceParametrizer(nn.Module):
    def __init__(self):
        super(RelevanceParametrizer, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5,5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5,5))
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        out = F.max_pool2d(F.relu(self.conv2(out)), (2,2))
        out = out.view(-1, num_flat_features(out))
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out


class Aggregator(nn.Module):
    def __init__(self):
        super(Aggregator, self).__init__()

    def forward(self, theta, h):
        g = theta + h
        return g


class SENN(nn.Module):
    def __init__(self):
        super(SENN, self).__init__()
        self.concept_encoder = ConceptEncoder()
        self.relevance_parametrizer = RelevanceParametrizer()
        self.aggregator = Aggregator()

    def forward(self, x):
        ## relevance parametrizer
        theta = self.relevance_parametrizer(x)

        ## concept encoder
        h = self.concept_encoder(x)

        ## aggregator
        g = self.aggregator(theta, h)
        return g, h, theta


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = SENN().to(device)

    inp = torch.rand((2,1,28,28)).to(device)
    g, h, theta = model(inp)

    print(g.shape)
    print(h.shape)
    print(theta.shape)

    # print(list(model.aggregator.parameters()))
