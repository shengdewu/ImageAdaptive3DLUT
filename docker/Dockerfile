FROM dl.nvidia/cuda:10.2-cudnn7-devel-torch1.10

COPY ./ImageAdaptive3DLUT /code/LUT
RUN  rm -rf .git && rm -rf .idea && cd /code/LUT/trilinear/cpp rm -rf build && rm -rf trilinear.* && sh setup.torch1.x.sh

WORKDIR /code
ENTRYPOINT ["python3", "LUT/main.py"]
