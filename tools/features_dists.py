from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns
from tqdm import tqdm
import numpy as np
import itertools
import argparse
import random
import json

from vissl.utils.extract_features_utils import ExtractedFeaturesLoader


def load_data(root_path: Path, layer: str = "heads", subsample: int = -1,
    classes: list = None, split: str = "test", seed: int = 0):
    data = ExtractedFeaturesLoader.sample_features(
        input_dir=root_path,
        split=split,
        layer=layer,
        flatten_features=False,
        num_samples=subsample,
        seed=seed
    )
    
    x = data["features"]
    y = data["targets"].flatten()

    if classes is not None:
        x = x[np.isin(y, classes)]
        y = y[np.isin(y, classes)]

    print(f"Features dim: {x.shape}, classes: {set(list(y))}")
    return x, y


def mean_distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    
    features_id = {
        int(i): x[y==target] for i, target in enumerate(np.unique(y))
    }

    mean_distances = np.empty((len(features_id), len(features_id)))
    for i in tqdm(features_id):
        for j in features_id:
            distances = []
            for va in features_id[i]:
                for vb in features_id[j]:
                    distances.append(np.linalg.norm(va-vb))
            mean_distances[i,j] = np.array(distances).mean()

    print("Distance Matrix:")
    print(mean_distances)
    
    return mean_distances


def plot_distance_matrix(dmat: np.ndarray, out_path: Path, title: str = ""):
    # TODO: set class names
    
    sns.set_style("whitegrid", {'axes.grid' : False})

    s = max(10, len(dmat) * 2)
    plt.figure(figsize=(s, s))
    if title:
        plt.title(title)

    im = plt.imshow(dmat, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    tick_marks = np.arange(len(dmat))
    plt.xticks(tick_marks, range(len(dmat)), rotation=45, ha='right')
    plt.yticks(tick_marks, range(len(dmat)))
    
    thresh = ((dmat.max()-dmat.min()) / 2) + dmat.min()
    for i, j in itertools.product(range(dmat.shape[0]), range(dmat.shape[1])):
        plt.text(j, i, str(dmat[i, j].round(5)), horizontalalignment="center", 
            color="white" if dmat[i, j] > thresh else "black")
    fig = plt.gcf()
    fig.savefig(out_path, dpi=100)


def class_names_to_index(index_map_file:Path, class_names: list) -> list:
    with open(index_map_file, "r") as f:
        labels = json.load(f)
    class_idx = [labels[i] for i in class_names]
    return class_idx


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""Computes a distance matrix between all the feature
            vectors of each class"""
    )

    parser.add_argument(
        "features",
        help="Root path to the features",
        type=Path
    )
    parser.add_argument(
        "-l",
        "--layer",
        help="Features layer. Default 'heads'",
        default="heads"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output image",
        type=Path,
        required=False
    )
    parser.add_argument(
        "--title",
        required=False,
        help="Plot title"
    )
    parser.add_argument(
        "--subsample",
        help="Take only n random samples from each class",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--classes",
        help="List of classes to process separated by ','",
        required=False
    )
    parser.add_argument(
        "--label-map",
        help="Path to the 'label_to_index_map.json'. Default to None",
        default=None,
        type=Path
    )
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        help="Random seed"
    )

    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)

    if args.classes is not None:
        classes = class_names_to_index(args.label_map, args.classes.split(","))
    else:
        classes = None
    
    x, y = load_data(root_path=args.features, layer=args.layer, 
        subsample=args.subsample, classes=classes, seed=args.seed)

    dmat = mean_distance_matrix(x, y)
    
    if args.output is not None:
        plot_distance_matrix(dmat, out_path=args.output, title=args.title)
