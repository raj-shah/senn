import torch
import torch.nn as nn
import torch.nn.functional as F

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class ConceptAutoencoder(nn.Module):
    def __init__(self, num_concepts):
        super(ConceptAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(True),
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Linear(64, 12), nn.ReLU(True),
            nn.Linear(12, num_concepts)
        )

        self.decoder = nn.Sequential(
            nn.Linear(num_concepts, 12), nn.ReLU(True),
            nn.Linear(12, 64), nn.ReLU(True),
            nn.Linear(64, 128), nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder


## RelevanceParametrizer == LeNet
class RelevanceParametrizer(nn.Module):
    def __init__(self, num_concepts):
        super(RelevanceParametrizer, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5,5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5,5))
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_concepts)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.maxpool(self.relu(self.conv1(x)))
        out = self.maxpool(self.relu(self.conv2(out)))
        out = out.view(-1, num_flat_features(out))
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out


class Aggregator(nn.Module):
    def __init__(self):
        super(Aggregator, self).__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 10)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, concept, relevance):
        out = concept + relevance
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class SENN(nn.Module):
    def __init__(self, num_concepts=5):
        super(SENN, self).__init__()
        self.concept_autoencoder = ConceptAutoencoder(num_concepts)
        self.relevance_parametrizer = RelevanceParametrizer(num_concepts)
        self.aggregator = Aggregator()

    def forward(self, x):
        ## concept encoder
        concept_encoder, concept_decoder = self.concept_autoencoder(x.view(x.size(0), -1))

        ## relevance parametrizer
        relevance = self.relevance_parametrizer(x)

        ## aggregator
        out = self.aggregator(concept_encoder, relevance)

        return concept_encoder, concept_decoder, relevance, out


if __name__ == '__main__':
    model = SENN(num_concepts=5)
    inp = torch.rand((2,1,28,28))
    model(inp)
