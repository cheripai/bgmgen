import numpy as np
import pickle
import sys
import torch
from models.network import Encoder, Generator
from torch.autograd import Variable
from tqdm import tqdm
from utils import song_to_pianoroll, write_pianoroll

IMG_FEAT_SIZE = 512
HIDDEN_SIZE = 256
MAX_LENGTH = 768
ENCODER_WEIGHTS_PATH = "data/encoder.pth"
GENERATOR_WEIGHTS_PATH = "data/generator.pth"

use_cuda = torch.cuda.is_available()

def generate(x, encoder, generator, tokens):
    batch_size = 1

    img_feats = encoder(x)

    hidden = generator.init_hidden(batch_size)
    hidden = hidden.cuda() if use_cuda else hidden

    generator_input = Variable(torch.LongTensor(batch_size, 1))
    generator_input[:] = tokens.index("SOS")
    generator_input = generator_input.cuda() if use_cuda else generator_input

    song = Variable(torch.LongTensor(MAX_LENGTH))
    for i in range(MAX_LENGTH):
        output, hidden = generator(generator_input, img_feats, hidden)
        _, top = output.data.topk(1)
        generator_input = Variable(torch.LongTensor(batch_size, 1))
        generator_input = generator_input.cuda() if use_cuda else generator_input
        generator_input[:] = top
        song[i] = top
        
    return song.data.cpu().numpy()


if __name__ == "__main__":
    bgs = np.load(sys.argv[1])
    tokens = pickle.load(open(sys.argv[2], "rb"))

    encoder = Encoder(IMG_FEAT_SIZE)
    generator = Generator(IMG_FEAT_SIZE, HIDDEN_SIZE, len(tokens), n_layers=2)
    encoder.load_state_dict(torch.load(ENCODER_WEIGHTS_PATH))
    generator.load_state_dict(torch.load(GENERATOR_WEIGHTS_PATH))

    if use_cuda:
        encoder = encoder.cuda()
        generator = generator.cuda()

    for i in tqdm(range(len(bgs))):
        bg = Variable(torch.from_numpy(bgs[i]).type(torch.FloatTensor))

        if use_cuda:
            bg = bg.cuda()

        # TEMP
        bg = bg.unsqueeze(0)

        song = generate(bg, encoder, generator, tokens)
        pianoroll = song_to_pianoroll(song.data, tokens)
        write_pianoroll(pianoroll, "results/{}.midi".format(i))
