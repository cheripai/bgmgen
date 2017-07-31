import bcolz
import numpy as np
import os
import sys
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.preprocessing import image
from random import shuffle

MAX_LENGTH = 16384


def img2feat(img, model):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x)


if __name__ == "__main__":
    img_dir = sys.argv[1]
    pianoroll_dir = sys.argv[2]

    base_model = VGG16(include_top=True, weights="imagenet")
    model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc2").output)

    img_paths = []

    for root, dirs, files in os.walk(img_dir):
        for f in files:
            if f.endswith(".jpg"):
                img_paths.append(os.path.join(root, f))

    pianoroll_paths = [img_path.replace(img_dir, pianoroll_dir, 1).replace(".jpg", ".bcolz") for img_path in img_paths]

    N = len(img_paths)

    index_shuffled = list(range(N))
    shuffle(index_shuffled)

    feats = np.zeros((N, 4096))
    for img_path, i in zip(img_paths, index_shuffled):
        img = image.load_img(img_path, target_size=(224, 224))
        feats[i] = img2feat(img, model)
    
    pianorolls = np.zeros((N, MAX_LENGTH, 128))
    for pianoroll_path, i in zip(pianoroll_paths, index_shuffled):
        pianoroll = bcolz.open(pianoroll_path)[:]
        pianoroll_length = min(MAX_LENGTH, pianoroll.shape[0])
        pianorolls[i,:pianoroll_length,:] = pianoroll[:pianoroll_length,:]

    c = bcolz.carray(feats, rootdir="img_feats.bcolz", mode="w")
    c.flush()
    c = bcolz.carray(pianorolls, rootdir="pianorolls.bcolz", mode="w")
    c.flush()
