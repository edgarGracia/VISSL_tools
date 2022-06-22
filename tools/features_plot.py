from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np
import argparse
import random
import json

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from vissl.utils.extract_features_utils import ExtractedFeaturesLoader

sns.set(rc={'figure.figsize':(11.7,8.27)})



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


def plot_tsne(x: np.ndarray, y: np.ndarray, out_path: Path = None,
    scale: bool = True, title: str = None, label_map: dict = None):
    
    palette = sns.color_palette("colorblind", len(set(y)))

    if scale:
        x = StandardScaler().fit_transform(x)

    tsne = TSNE()
    X_embedded = tsne.fit_transform(x)

    if label_map is not None:
        y = [label_map[i] for i in y]
    
    plot = sns.scatterplot(
        X_embedded[:,0],
        X_embedded[:,1],
        hue=y,
        legend='full',
        palette=palette
    )

    if title is not None:
        plot.set_title(title)

    if out_path is not None:
        plt.savefig(str(out_path))
    else:
        plt.show()
    plt.close()
    

def plot_umap(x: np.ndarray, y: np.ndarray, out_path: Path = None,
    scale: bool = True, title: str = None, label_map: dict = None):
    
    import umap # pip install umap-learn
    
    palette = sns.color_palette("colorblind", len(set(y)))
    
    if scale:
        x = StandardScaler().fit_transform(x)
    
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(x)
    
    if label_map is not None:
        y = [label_map[i] for i in y]

    plot = sns.scatterplot(
        embedding[:,0],
        embedding[:,1],
        hue=y,
        legend='full',
        palette=palette
    )
    
    if title is not None:
        plot.set_title(title)

    if out_path is not None:
        plt.savefig(str(out_path))
    else:
        plt.show()
    plt.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Apply a tsne or a umap over vissl features")

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
        "--scale",
        help="Automatic scaling of the features",
        action="store_true"
    )
    parser.add_argument(
        "--tsne",
        action="store_true"
    )
    parser.add_argument(
        "--umap",
        action="store_true"
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
    parser.add_argument(
        "--set",
        default="test",
        help='Data set. "train" or "test". Default to "test"'
    )

    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)

    # Get class names
    id_labels, classes = None, None
    if args.label_map is not None:
        with open(args.label_map, "r") as f:
            labels_id = json.load(f)
        id_labels = {v:k for k,v in labels_id.items()}
        
        # Filter classes to plot
        if args.classes is not None:
            classes = [labels_id[i] for i in args.classes.split(",")]
    
    x, y = load_data(root_path=args.features, layer=args.layer, 
        subsample=args.subsample, classes=classes, seed=args.seed,
        split=args.set)

    if args.tsne:
        out = (args.output.parent.joinpath("tsne_" + args.output.name)
            if args.umap and args.output is not None else args.output)
        plot_tsne(x, y, out_path=out, scale=args.scale, title=args.title,
            label_map=id_labels)
    if args.umap:
        out = (args.output.parent.joinpath("umap_" + args.output.name)
            if args.tsne and args.output is not None else args.output)
        plot_umap(x, y, out_path=out, scale=args.scale, title=args.title,
            label_map=id_labels)
