import sklearn.metrics as skm
from pathlib import Path
import numpy as np
import argparse
import json

from vissl.utils.extract_features_utils import ExtractedFeaturesLoader

from plot_confusion_matrix import plot_confusion_matrix



def load_data(root_path: Path, layer: str = "heads", subsample: int = -1,
    classes: list = None, split: str = "test", seed: int = 0):
    data = ExtractedFeaturesLoader.load_features(
        input_dir=root_path,
        split=split,
        layer=layer,
        flatten_features=False,
    )
    
    scores = data["features"]
    targets = data["targets"].flatten()

    print(f"Score dim: {scores.shape}")
    return scores, targets


def evaluate_features(scores: np.ndarray, targets: np.ndarray,
    label_map: list = None, output_path: Path = None, title: str = ""):

    x = scores.argmax(1)

    if label_map is None:
        target_names = [str(i) for i in range(len(set(list(targets))))]
    else:
        target_names = list(label_map.keys())
    
    print("Target names:", target_names)

    cm = skm.confusion_matrix(targets, x)
    score = skm.classification_report(targets, x, target_names=target_names)
    
    print("\n---------- Confusion Matrix ----------")
    print(cm)
    print("\n-------- Classification Report --------")
    print(score)

    if output_path is not None:
        with open(output_path.joinpath("classification_report.txt"), "w") as f:
            f.write(str(cm)+"\n")
            f.write(score)

        plot_confusion_matrix(
            cm,
            target_names=target_names,
            output_path=output_path.joinpath("confusion_matrix.png"),
            title=title
        )



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate the output of a classifier model")

    parser.add_argument(
        "features",
        help="Root path to the features",
        type=Path
    )
    parser.add_argument(
        "--label-map",
        help="Path to the 'label_to_index_map.json'. Default to None",
        default=None,
        type=Path
    )
    parser.add_argument(
        "-l",
        "--layer",
        help="Features layer. Default 'heads'",
        default="heads"
    )
    parser.add_argument(
        "-s",
        "--split",
        help="'test' or 'train'",
        default="test"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output folder where save metrics and the confusion matrix",
        type=Path,
        default=None
    )
    parser.add_argument(
        "--title",
        help="Confusion matrix title",
        default=""
    )

    args = parser.parse_args()

    if args.label_map is not None:
        with open(args.label_map, "r") as f:
            labels = json.load(f)
    else:
        labels = None

    if args.output is not None:
        args.output.mkdir(exist_ok=True, parents=True)

    scores, targets = load_data(
        root_path=args.features,
        split=args.split,
        layer=args.layer
    )
    evaluate_features(
        scores=scores,
        targets=targets,
        label_map=labels,
        output_path=args.output,
        title=args.title
    )
