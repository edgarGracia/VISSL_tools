# VISSL_tools

## Train a custom dataset on a single GPU
1. Copy the content of ```./configs``` to ```~/vissl/configs/config/my/```
2. Copy the content of ```./tools``` to ```~/vissl/tools/my/```
4. Download the [RN50](https://download.pytorch.org/models/resnet50-19c8e357.pth) pretrained weights to ```~/vissl/pretrained_models/```
5. Open ```~/vissl/configs/config/dataset_catalog.json``` and add the paths to the datasets
6. Train SimCLR-resnet50 with pretrained weights:
```
python tools/run_distributed_engines.py hydra.verbose=true config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true config.HOOKS.MEMORY_SUMMARY.PRINT_MEMORY_SUMMARY=false config=pretrain/simclr/simclr_8node_resnet config.CHECKPOINT.DIR=/train/path config.DATA.TRAIN.DATASET_NAMES=[dataset_name] config.DATA.TEST.DATASET_NAMES=[dataset_name] config.DISTRIBUTED.NUM_NODES=1 config.DISTRIBUTED.NUM_PROC_PER_NODE=1 config.DISTRIBUTED.RUN_ID=auto config.MODEL.WEIGHTS_INIT.PARAMS_FILE=pretrained_models/resnet50-19c8e357.pth config.MODEL.WEIGHTS_INIT.APPEND_PREFIX="trunk._feature_blocks." config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=""
```
5. Extract features
```
python tools/run_distributed_engines.py config=my/resnet_feature_extraction config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/train/path/checkpoint.torch config.EXTRACT_FEATURES.OUTPUT_DIR=/features/output/path config.DATA.TRAIN.DATASET_NAMES=[dataset_name] config.DATA.TEST.DATASET_NAMES=[dataset_name]
```
6. Visualize features
```
python tools/my/features_plot.py /path/to/features --scale --umap --title a_tile -o umap.png
```
7. Train a supervised classifier
- Set the number of classes on the config file ```~/vissl/configs/config/my/simclr_resnet_transfer_linear_single_head.yaml``` (line 70)
```
python tools/run_distributed_engines.py hydra.verbose=true config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true config.HOOKS.MEMORY_SUMMARY.PRINT_MEMORY_SUMMARY=false config=my/simclr_resnet_transfer_linear_single_head config.CHECKPOINT.DIR=/classifier/save/train/path/ config.DATA.TRAIN.DATASET_NAMES=[dataset_name] config.DATA.TEST.DATASET_NAMES=[dataset_name] config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/previous/train/path/checkpoint.torch
```
8. Extract features of the classifier model
- Set the number of classes on the config file ```~/vissl/configs/config/my/simclr_resnet_transfer_linear_single_head_feature_extraction.yaml``` (line 66)
```
python tools/run_distributed_engines.py config=my/simclr_resnet_transfer_linear_single_head_feature_extraction config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/train/path/checkpoint.torch config.EXTRACT_FEATURES.OUTPUT_DIR=/features/out/path config.DATA.TRAIN.DATASET_NAMES=[dataset_name] config.DATA.TEST.DATASET_NAMES=[dataset_name]
```
9. Evaluate the classified features
```
python tools/my/cls_evaluation.py /path/to/classified/features -l res5 -s test --title a_tile -o /output/evaluation/path --label-map /save/train/path/test_label_to_index_map.json
```
