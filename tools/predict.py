from pathlib import Path
from typing import Union
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse

from vissl.models import build_model
from vissl.data.ssl_transforms import get_transform
from classy_vision.generic.util import load_checkpoint
from vissl.utils.checkpoint import init_model_from_consolidated_weights
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict


class Predictor:

    def __init__(self, model_config: dict, weights_path: Union[str, Path]):

        self.model_config = model_config
        self.weights_path = str(weights_path)
        
        # Create the cfg
        self.cfg = [
            f'config={model_config}',
            f'config.MODEL.WEIGHTS_INIT.PARAMS_FILE={weights_path}',
        ]
        self.cfg = compose_hydra_configuration(self.cfg)
        _, self.cfg = convert_to_attrdict(self.cfg)

        # Create transforms
        split = "TEST" if "TEST" in self.cfg["DATA"] else "TRAIN"
        self.transform = get_transform(self.cfg["DATA"][split].TRANSFORMS)

        # Load the model
        model = build_model(self.cfg.MODEL, self.cfg.OPTIMIZER)
        weights = load_checkpoint(
            checkpoint_path=self.cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)
        init_model_from_consolidated_weights(
            config=self.cfg,
            model=model,
            state_dict=weights,
            state_dict_key_name="classy_state_dict",
            skip_layers=[],
        )
        
        model.eval()
        self.model = model


    def predict(self, image: Union[str, Path, Image.Image]):
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # Convert images to RGB. This is important
        image = image.convert("RGB")

        # Image transform
        x = self.transform({"data": [image]})["data"][0]

        # Predict
        features = self.model(x.unsqueeze(0))
        features = features[0].cpu().detach().numpy()

        return features


def predict_images_on_path(predictor: Predictor, in_path: Path,
    out_path: Path = None):
    """Predict an image or all the images on a path recursively and saves the
    output on .npy files.

    Args:
        predictor (Predictor): A Predictor object.
        in_path (Path): Input images path or a single image.
        out_path (Path, optional): Path where save the output of the model.
          If None only prints the output. Defaults to None.
    """
    
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    if in_path.is_file():
        images = [in_path]
    else:
        images = list(in_path.iterdir())

    pbar = tqdm(images, disable=out_path is None)
    for img_path in pbar:
        pbar.set_description(str(img_path))
        if img_path.is_file():
            pred = predictor.predict(img_path)
            if out_path is not None:
                np.save(out_path / (img_path.stem + ".npy"), pred)
            else:
                print(str(img_path),"\n",pred)
        else:
            predict_images_on_path(predictor, img_path,
                out_path / img_path.name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default='pretrain/simclr/simclr_8node_resnet.yaml',
        type=Path,
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=Path,
        required=True
    )
    parser.add_argument(
        "-i",
        "--images",
        help="Path to an image or a folder with images",
        type=Path,
        required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output path to save the predictions",
        type=Path,
        required=False
    )

    args = parser.parse_args()

    predictor = Predictor(
        model_config=args.config,
        weights_path=args.weights,
    )

    predict_images_on_path(predictor, args.images, args.output)
