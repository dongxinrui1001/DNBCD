# BUSI classification project

This is a PyCharm friendly version of the BUSI experiment in
`model-comparison-for-x-ai-based-dnbcd-busi.ipynb`. It keeps the original
TensorFlow/Keras pipeline and removes the Kaggle-only paths and repeated cells.

## Dataset layout

Set `--data` to the folder containing:

```text
Dataset_BUSI_with_GT/
├── benign/
├── malignant/
└── normal/
```

BUSI mask files whose names contain `_mask` are excluded from classification.

## Run from PyCharm

Set the interpreter to `D:\anaconda_envs\dnbcd\python.exe`, mark this folder as
the Sources Root, and run `train.py`.

```bash
python train.py --data "F:\daimaFuxian\DNBCD\data\Dataset_BUSI_with_GT" --model dnbcd --epochs 20
```

Available models: `dnbcd`, `densenet`, `mobilenet`, `resnet50`, `vgg19`.
Outputs are written to `runs/busi/`: split data, models, CSV logs, metrics, and a confusion matrix.
