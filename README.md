# UltraDINO: Code release for "General methods make great domain-specific foundation models" (MICCAI25)

This repository contains the code to reproduce experiments of the MICCAI 2025 paper "General methods make great domain-specific foundation models". The codebase is largely based on DINOv2, with minor modifications to pretraining (Greyscale inputs, tensorboard logging), and new finetuning implementations for classification and segmentation.

## Installation

* Dependencies: Python 3.11

In a virtual environment, run:

```
pip install -e .
```

## Classification

Classification code is based on a custom pytorch lightning + hydra implementation.

Run training:

```
python classification/train.py --config-name fine_tune_barcelona_vitb14
```

run test:

```
python classification/test.py --config-name fine_tune_barcelona_vitb14 test.checkpoint_path="PATH/TO/FINETUNED/WEIGHTS.pth
```

Note that you may need to escape `=` in any checkpoint paths by changing to `\=`, otherwise the path interferes with omegaconf.

## Segmentation

For segmentation finetuning, we rely on mmseg.

run training:
```
python segmentation/train.py [[CONFIG_FILE]]
```

run test:
```
python segmentation/test.py [[CONFIG_FILE]] [[CHECKPOINT_FILE]]
```



## Frequent issues

###  mmcv._ext not found

First uninstall any existing versions of mmcv, then install mmcv using

```
pip install mmcv==2.1.0 --no-cache-dir --no-binary :all:
```

## Citation

If you are using this code, please consider citing:

```
@inproceedings{ambsdorf2025general,
  title={General methods make great domain-specific foundation models: A case-study on fetal ultrasound},
  author={Ambsdorf, Jakob and Munk, Asbj{\o}rn and Llambias, Sebastian and Christensen, Anders N and Mikolaj, Kamil and Balestriero, Randall and Tolsgaard, Martin G and Feragen, Aasa and Nielsen, Mads},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={271--281},
  year={2025},
  organization={Springer}
}
```