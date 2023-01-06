# 训练配置
### 基础配置
```python
DATALOADER:
  DATASET: ImageDataSetXinTu
  DATA_PATH: /home/shengdewu/data/xt.image.enhancement.540
  NUM_WORKERS: 10
  XT_TEST_MAX_NUMS: 60
  XT_TEST_TXT: no_aug.test.txt
  XT_TRAIN_INPUT_TXT: no_aug.train_input.txt
  XT_TRAIN_LABEL_TXT: no_aug.train_label.txt
INPUT:
  COLOR_JITTER:
    ADAPTIVE_LIGHT:
      ENABLED: false
      MAX: 1.2
      MIN: 0.8
    BRIGHTNESS:
      ENABLE: false
      MAX: 1.05
      MIN: 0.7
    CONTRAST:
      ENABLE: false
      MAX: 1.1
      MIN: 0.8
    PROB: 0.0
    SATURATION:
      ENABLE: false
      MAX: 1.1
      MIN: 0.7
  INPUT_OVER_EXPOSURE:
    ENABLED: false
    F_MAX: -0.08
    F_MIN: -0.4
    F_VALUE: 1.5
  TRAINING_COLOR_JITTER:
    BRIGHTNESS:
      MAX: 1.2
      THRESHOLD: 0.3
    CONTRAST: 1.1
    DARKNESS:
      MIN: 0.85
      THRESHOLD: 0.7
    ENABLE: false
    SATURATION: 1.0
MODEL:
  ARCH: AdaptivePerceptualPairedModel
  CLASSIFIER:
    ARCH: MobileNet
  DEVICE: cuda
  LUT:
    DIMS: 16
    SUPPLEMENT_NUMS: 11
    ZERO_LUT: true
  VGG:
    VGG_LAYER: 14
    VGG_PATH: /mnt/sdb/pretrain.model/vgg.model/pytorch/vgg16-397923af.pth
  WEIGHTS: ''
OUTPUT_DIR: /mnt/sda1/train.output/enhance.output/img.lut12.mobile.dim16
OUTPUT_LOG_NAME: image.lut
SOLVER:
  BASE_LR: 0.0002
  WARMUP_ITERS: 100  # or  0
  GAMMA: 0.1
  MAX_ITER: 200000
  STEPS: (15000, 19000)
  CHECKPOINT_PERIOD: 5000
  MAX_KEEP: 10
  IMS_PER_BATCH: 8
  TEST_PER_BATCH: 8
  LAMBDA_GP: 10
  LAMBDA_PIXEL: 6  # for AdaptiveUnPairedModel 1000
#  """
#  1.the LAMBDA_SMOOTH should be small since a heavy smoothness penalty will result in flat 3D LUTs while limited transformation flexibility
#  2. the LAMBDA_MONOTONICITY , it can be relatively stronger, can help to update the parameters that may be not be activated by input data
#  3. the LAMBDA_SMOOTH more than 0.0001 leads to worse PSNR , but PSNR is insensitive to the choice LAMBDA_MONOTONICITY
#  4. base LAMBDA_SMOOTH and LAMBDA_MONOTONICITY (0.0001, 10)
#  """
  LAMBDA_CLASS_SMOOTH: 5.0e-05  #正常情况 等于 LAMBDA_SMOOTH
  LAMBDA_SMOOTH: 1  # 0 0.00001 0.0001 0.001 0.01 0.1
  LAMBDA_MONOTONICITY: 30 # 0.1 0 1.0 10 100 1000
  LAMBDA_PERCEPTUAL: 0.05  # value similar LAMBDA_SMOOTH
  N_CRITIC: 1
  ADAM:
    B1: 0.9
    B2: 0.999
```

### 微调（此过程的数据增强在 data.imbalance/data_imbalance.py 中)
1. 自定义过曝欠曝数据集
* add_over_expose_by_gt()
* add_under_expose_by_gt()
* split_only_over_under_expose_by_gt()
```python
DATALOADER:
  DATASET: ImageDataSetXinTu
  DATA_PATH: /home/shengdewu/data/xt.image.enhancement.540
  NUM_WORKERS: 8
  XT_TEST_MAX_NUMS: 100
  XT_TEST_TXT: over_under.test.txt
  XT_TRAIN_INPUT_TXT: over_under.train_input.txt
  XT_TRAIN_LABEL_TXT: over_under.train_label.txt
INPUT:
  COLOR_JITTER:
    ADAPTIVE_LIGHT:
      ENABLED: false
      MAX: 1.2
      MIN: 0.8
    BRIGHTNESS:
      ENABLE: false
      MAX: 1.1
      MIN: 0.8
    CONTRAST:
      ENABLE: false
      MAX: 1.1
      MIN: 0.8
    PROB: 0.0
    SATURATION:
      ENABLE: false
      MAX: 1.1
      MIN: 0.7
  INPUT_OVER_EXPOSURE:
    ENABLED: false
    F_MAX: -0.08
    F_MIN: -0.4
    F_VALUE: 1.5
  TRAINING_COLOR_JITTER:
    BRIGHTNESS:
      MAX: 1.2
      THRESHOLD: 0.3
    CONTRAST: 1.1
    DARKNESS:
      MIN: 0.85
      THRESHOLD: 0.7
    ENABLE: false
    SATURATION: 1.0
MODEL:
  ARCH: AdaptivePairedModel
  CLASSIFIER:
    ARCH: MobileNet
    ROUGH_SIZE: 540
  DEVICE: cuda
  LUT:
    DIMS: 16
    SUPPLEMENT_NUMS: 11
    ZERO_LUT: true
  WEIGHTS: /mnt/sda1/train.output/enhance.output/img.lut12.mobile.dim16/AdaptivePerceptualPairedModel_final.pth
OUTPUT_DIR: /mnt/sda1/train.output/enhance.output/lut.pretrain
OUTPUT_LOG_NAME: image.lut
SOLVER:
  BASE_LR: 0.00001
  WARMUP_ITERS: 100
  GAMMA: 0.1
  MAX_ITER: 100000
  STEPS: (40000, 80000)
  CHECKPOINT_PERIOD: 5000
  MAX_KEEP: 10
  IMS_PER_BATCH: 8
  TEST_PER_BATCH: 2
#  LAMBDA_GP: 10
  LAMBDA_PIXEL: 100  # for AdaptiveUnPairedModel 1000
#  """
#  1.the LAMBDA_SMOOTH should be small since a heavy smoothness penalty will result in flat 3D LUTs while limited transformation flexibility
#  2. the LAMBDA_MONOTONICITY , it can be relatively stronger, can help to update the parameters that may be not be activated by input data
#  3. the LAMBDA_SMOOTH more than 0.0001 leads to worse PSNR , but PSNR is insensitive to the choice LAMBDA_MONOTONICITY
#  4. base LAMBDA_SMOOTH and LAMBDA_MONOTONICITY (0.0001, 10)
#  """
  LAMBDA_CLASS_SMOOTH: 5.0e-05  #正常情况 等于 LAMBDA_SMOOTH
  LAMBDA_SMOOTH: 1  # 0 0.00001 0.0001 0.001 0.01 0.1
  LAMBDA_MONOTONICITY: 30 # 0.1 0 1.0 10 100 1000
  # LAMBDA_PERCEPTUAL: 0.05  # value similar LAMBDA_SMOOTH
  N_CRITIC: 1
  ADAM:
    B1: 0.9
    B2: 0.999
```

2. 通过PS制作过曝欠曝数据集
* select_class_img()
* split_ps_data()
```python

  XT_TEST_TXT: ps.test.txt
  XT_TRAIN_INPUT_TXT: ps.train_input.txt
  XT_TRAIN_LABEL_TXT: ps.train_label.txt

```