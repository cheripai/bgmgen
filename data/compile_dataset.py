import bcolz
import numpy as np
import os
import sys
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from random import shuffle


def img2feat(img, model):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feat = model.predict(x)


if __name__ == "__main__":
    img_dir = sys.argv[1]
    pianoroll_dir = sys.argv[2]

    model = VGG16(include_top=False, weights="imagenet")

    img_paths = []

    for root, dirs, files in os.walk(img_dir):
        for f in files:
            if f.endswith(".jpg"):
                img_paths.append(os.path.join(root, f))

    pianoroll_paths = [img_path.replace(img_dir, pianoroll_dir, 1).replace(".jpg", ".bcolz") for img_path in img_paths]

    feats = np.zeros((len(img_paths), 224*224))
    for i, img_path in enumerate(img_paths):
        img = image.load_img(img_path, target_size=(224, 224))
        feats[i] = img2feat(img, model)
    
    pianorolls = []
    for pianoroll_path in pianoroll_paths:
        pianoroll = bcolz.open(pianoroll_path)[:]
        pianorolls.append(pianoroll)

    index_shuffled = list(range(len(pianorolls)))
    shuffle(index_shuffled)
    feats_shuffled = np.zeros((len(pianorolls), 224*224))
    pianorolls_shuffled = []
    for i, index in enumerate(index_shuffled):
        feats_shuffled[i] = feats[index]
        pianorolls_shuffled.append(pianorolls[index])

    c = bcolz.carray(feats_shuffled, rootdir="img_feats.bcolz", mode="w")
    c.flush()
    c = bcolz.carray(pianorolls_shuffled, rootdir="pianorolls.bcolz", mode="w")
    c.flush()
