import bcolz
import random
import sys
import torch
from models.network import Generator
from torch.autograd import Variable

from tqdm import tqdm

IMG_FEAT_SIZE = 4096
OUTPUT_SIZE = 128
HIDDEN_SIZE = 512
epochs = 1

use_cuda = torch.cuda.is_available()


def train(x, y, generator, opt, criterion, teacher_forcing_ratio=0.5, train=False):
    opt.zero_grad()

    batch_size = 1
    hidden = generator.init_hidden(batch_size)
    hidden = hidden.cuda() if use_cuda else hidden
    max_length = y.size(1)

    if train:
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    else:
        use_teacher_forcing = False

    generator_input = Variable(torch.FloatTensor(batch_size, OUTPUT_SIZE))
    generator_input[:] = 0
    generator_input = generator_input.cuda() if use_cuda else generator_input

    loss = 0

    if use_teacher_forcing:
        for i in range(max_length):
            if y[:, i].sum().data[0] < 0:
                break
            output, hidden = generator(generator_input, x, hidden)
            loss += criterion(output, y[:, i])
            generator_input[:] = y[:, i]

    else:
        for i in range(max_length):
            if y[:, i].sum().data[0] < 0:
                break
            output, hidden = generator(generator_input, x, hidden)
            loss += criterion(output, y[:, i])
            generator_input[:] = output

    if train:
        loss.backward()
        opt.step()

    return loss.data[0] / i


if __name__ == "__main__":
    imgs = bcolz.open(sys.argv[1])
    pianorolls = bcolz.open(sys.argv[2])

    generator = Generator(IMG_FEAT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    criterion = torch.nn.MultiLabelMarginLoss()

    if use_cuda:
        generator = generator.cuda()
        criterion = criterion.cuda()

    N = len(imgs)
    for e in range(epochs):
        shuffled_indexes = list(range(N))
        random.shuffle(shuffled_indexes)
        epoch_loss = 0
        for i in tqdm(shuffled_indexes):
            img, pianoroll = imgs[i], pianorolls[i]
            img = Variable(torch.from_numpy(img).type(torch.FloatTensor))
            pianoroll = Variable(torch.from_numpy(pianoroll).type(torch.LongTensor))

            if use_cuda:
                img = img.cuda()
                pianoroll = pianoroll.cuda()

            # TEMP
            img = img.unsqueeze(0)
            pianoroll = pianoroll.unsqueeze(0)

            epoch_loss += train(img, pianoroll, generator, optimizer, criterion, train=True)
        print("Epoch {}: {}".format(e, epoch_loss / N))

        torch.save(generator.state_dict(), "data/weights.pth")

        
