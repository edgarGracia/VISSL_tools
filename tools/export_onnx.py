from pathlib import Path
import argparse

import torch

from vissl.models import build_model
from vissl.data.ssl_transforms import get_transform
from classy_vision.generic.util import load_checkpoint
from vissl.utils.checkpoint import init_model_from_consolidated_weights
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default='my/resnet_feature_extraction.yaml',
        type=Path,
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output onnx file",
        type=Path,
        required=True
    )

    args = parser.parse_args()

    # Create the cfg
    cfg = [
        f'config={str(args.config)}',
        f'config.MODEL.WEIGHTS_INIT.PARAMS_FILE={str(args.weights)}',
    ]
    cfg = compose_hydra_configuration(cfg)
    _, cfg = convert_to_attrdict(cfg)

    # Load the model
    model = build_model(cfg.MODEL, cfg.OPTIMIZER)
    weights = load_checkpoint(
        checkpoint_path=cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)
    init_model_from_consolidated_weights(
        config=cfg,
        model=model,
        state_dict=weights,
        state_dict_key_name="classy_state_dict",
        skip_layers=[],
    )
    
    model.eval()
    
    dummy_input = torch.randn(
        1, 3, args.input_size, args.input_size, device="cpu"
    )
    torch.onnx.export(model, dummy_input, args.output, verbose=True)
