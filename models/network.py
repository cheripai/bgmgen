import torch
import torch.nn as nn
from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self, img_feat_size, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=16384):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.linear = nn.Linear(img_feat_size + output_size, hidden_size)
        self.gru = nn.GRU(
            self.hidden_size,
            self.hidden_size,
            num_layers=n_layers,
            dropout=self.dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, img_feat, hidden):
        input_combined = torch.cat((input, img_feat), 1)
        output = self.dropout(self.linear(input_combined))
        output = output.unsqueeze(0)
        output, hidden = self.gru(output, hidden)
        output = self.sigmoid(self.out(output.squeeze(0)))
        return output, hidden

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(1, batch_size, self.hidden_size))


if __name__ == "__main__":
    hidden_size = 512
    output_size = 128
    img_feat_size = 4096
    batch = 4
    model = Generator(img_feat_size, hidden_size, output_size)

    x = Variable(torch.zeros(batch, output_size))
    y = Variable(torch.zeros(batch, img_feat_size))
    hidden = model.init_hidden(batch)
    
    output, hidden = model(x, y, hidden)
    for i in range(10):
        output, hidden = model(output, y, hidden)

    print(output.size(), hidden.size())
