FROM dl.nvidia/cuda:10.1-cudnn7-devel



COPY ./ImageAdaptive3DLUT /code/LUT
RUN cd /code/LUT/trilinear/cpp/torch_1_x && python3 setup.py build develop

cd trilinear/cpp/torch_1_x
sh setup.sh

COPY ./AdelaiDet /code/adet
RUN cd /code/adet python3 setup.py build develop


RUN pip3 install --no-cache-dir Pillow && \
    pip3 install --no-cache-dir fvcore && \
    pip3 install --no-cache-dir scikit-image && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache


WORKDIR /code

ENTRYPOINT ["python3", "adet/tools/train_net.py"]