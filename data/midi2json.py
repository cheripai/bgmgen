import os
import sys
from subprocess import call


def replace_root(path, target_dir):
    path = [target_dir] + path.split("/")[1:]
    return "/".join(path)


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
            if f.endswith(".mid"):
                sources.append(os.path.join(root, f))

    for source in sources:
        target = source.replace(".mid", ".json")
        target = replace_root(target, target_dir)
        targets.append(target)

    for source, target in zip(sources, targets):
        call(["node", "convert.js", source, target])
