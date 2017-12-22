import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, img_feat_size, dropout_p=0.1):
        super(Encoder, self).__init__()
        self.dropout_p = dropout_p

        self.conv1 = nn.Conv2d(3, 16, 3, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=2)
        self.bn5 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(8*8*256, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, img_feat_size)

        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, x):
        x = F.max_pool2d(self.bn1(F.relu(self.conv1(x))), (2, 2))
        x = F.max_pool2d(self.bn2(F.relu(self.conv2(x))), (2, 2))
        x = F.max_pool2d(self.bn3(F.relu(self.conv3(x))), (2, 2))
        x = F.max_pool2d(self.bn4(F.relu(self.conv4(x))), (2, 2))
        x = F.max_pool2d(self.bn5(F.relu(self.conv5(x))), (2, 2))
        x = x.view(-1, 8*8*256)
        x = self.dropout(self.bn6(F.relu(self.fc1(x))))
        x = self.fc2(x)
        return x
        

class Generator(nn.Module):
    def __init__(self, img_feat_size, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.linear = nn.Linear(img_feat_size + self.hidden_size, hidden_size)
        self.gru = nn.GRU(
            self.hidden_size,
            self.hidden_size,
            num_layers=n_layers,
            dropout=self.dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, img_feat, hidden):
        embedded = self.embedding(input).squeeze(1)
        embedded = self.dropout(embedded)
        output = embedded.unsqueeze(0)
        combined = torch.cat((embedded, img_feat), 1)
        output = self.dropout(self.linear(combined))
        output = output.unsqueeze(0)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output.squeeze(0)))
        return output, hidden

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(1, batch_size, self.hidden_size))


if __name__ == "__main__":
    batch_size = 4
    img_feat_size = 512

    model = Encoder(img_feat_size)

    img = Variable(torch.zeros((batch_size, 3, 224, 224)))
    model(img)
