FROM dl.nvidia/cuda:10.1-cudnn7-devel



COPY ./ImageAdaptive3DLUT /code/LUT
RUN cd /code/LUT/trilinear/cpp && sh setup.torch1.x.sh

RUN pip3 install --no-cache-dir Pillow && \
    pip3 install --no-cache-dir fvcore && \
    pip3 install --no-cache-dir scikit-image && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache


WORKDIR /code
ENTRYPOINT ["python3", "LUT/main.py"]