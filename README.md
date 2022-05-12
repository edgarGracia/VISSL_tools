# VISSL_tools

## Train a custom dataset on a single GPU
1. Copy config files to ```configs/config/my/```
2. Download pretrained weights from the VISSL repo to ```pretrain_models/```
3. Set the dataset paths on ```configs/config/dataset_catalog.json```
4. Train SimCLR-resnet50 with pretrained weights:
```
python tools/run_distributed_engines.py hydra.verbose=true config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true config.HOOKS.MEMORY_SUMMARY.PRINT_MEMORY_SUMMARY=false config=pretrain/simclr/simclr_8node_resnet config.CHECKPOINT.DIR=/save/train/path config.DATA.TRAIN.DATASET_NAMES=[dataset_name] config.DATA.TEST.DATASET_NAMES=[dataset_name] config.DISTRIBUTED.NUM_NODES=1 config.DISTRIBUTED.NUM_PROC_PER_NODE=1 config.DISTRIBUTED.RUN_ID=auto config.MODEL.WEIGHTS_INIT.PARAMS_FILE=pretrain_models/resnet50-19c8e357.pth config.MODEL.WEIGHTS_INIT.APPEND_PREFIX="trunk._feature_blocks." config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=""
```
5. Extract features
```
python tools/run_distributed_engines.py config=my/resnet_feature_extraction config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/save/train/path/checkpoint.torch config.EXTRACT_FEATURES.OUTPUT_DIR=/features/out/path config.DATA.TRAIN.DATASET_NAMES=[dataset_name] config.DATA.TEST.DATASET_NAMES=[dataset_name]
```
6. Visualize features
```
python features_plot.py /path/to/features --scale --tsne --title a_tile -o tsne.png
```
7. Train classifier
- Set num classes on the config file ```simclr_resnet_transfer_linear_single_head.yaml```
```
python tools/run_distributed_engines.py hydra.verbose=true config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true config.HOOKS.MEMORY_SUMMARY.PRINT_MEMORY_SUMMARY=false config=my/simclr_resnet_transfer_linear_single_head config.CHECKPOINT.DIR=/save/train/path/ config.DATA.TRAIN.DATASET_NAMES=[dataset_name] config.DATA.TEST.DATASET_NAMES=[dataset_name] config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/previous/simclr/train/path/checkpoint.torch
```
8. Extract features of the classifier model
```
python tools/run_distributed_engines.py config=my/simclr_resnet_transfer_linear_single_head_feature_extraction config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/train/path/checkpoint.torch config.EXTRACT_FEATURES.OUTPUT_DIR=/features/out/path config.DATA.TRAIN.DATASET_NAMES=[dataset_name] config.DATA.TEST.DATASET_NAMES=[dataset_name]
```
9. Evaluate the classified features
```
python cls_evaluation.py /path/to/classified/features -l res5 -s test --title a_tile -o /output/evaluation --label-map /save/train/path/test_label_to_index_map.json
```
