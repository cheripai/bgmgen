import bcolz
import json
import numpy as np
import os
import sys
from midi2json import replace_root
from pianoroll import Pianoroll


def preprocess(json):
    processed = {}
    try:
        i = 0
        while len(json["tracks"][i]["notes"]) == 0: i += 1
        processed["notes"] = json["tracks"][i]["notes"]
        processed["header"] = json["header"]
        return processed
    except:
        return None


if __name__ == "__main__":
    source_dir = sys.argv[1]
    target_dir = sys.argv[2]

    sources, targets = [], []
    for root, dirs, files in os.walk(source_dir):
        for d in dirs:
            target_d = os.path.join(replace_root(root, target_dir), d)
            if not os.path.exists(target_d):
                os.makedirs(target_d)
        for f in files:
            if f.endswith(".json"):
                sources.append(os.path.join(root, f))

    for source in sources:
        target = source.replace(".json", ".bcolz")
        target = replace_root(target, target_dir)
        targets.append(target)

    for source, target in zip(sources, targets):
        with open(source) as f:
            song = json.load(f)

        processed = preprocess(song)
        pianoroll = Pianoroll(processed["header"]["bpm"], processed["notes"])
        bcolz.carray(pianoroll.roll, rootdir=target).flush()
