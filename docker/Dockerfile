FROM dl.nvidia/cuda:11.1-cudnn8-devel-torch1.10.0

COPY  dataloader /code/lut/dataloader
COPY  engine /code/lut/engine
COPY  models /code/lut/models
COPY  schedule /code/lut/schedule
COPY  trilinear /code/lut/trilinear
COPY  trainer /code/lut/trainer
COPY  main.py /code/lut

RUN cd /code/lut/trilinear/cpp && rm -rf build && rm -rf trilinear.* && ls && sh setup.torch1.x.sh

WORKDIR /code
ENTRYPOINT ["python3", "lut/main.py"]
