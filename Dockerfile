FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
RUN apt-get update
RUN apt-get install -y python3.7-dev python3-pip wget libsndfile1

RUN mkdir /workspace
WORKDIR /workspace

COPY requirements.txt .

RUN python3.7 -m pip install -U pip setuptools
RUN python3.7 -m pip install -r requirements.txt

RUN mkdir ./model

RUN gdown --id "1c8xG8cysHslJRlgB5CYbUnWngEydIfI7" -O model/tacotron2-100k.h5
RUN gdown --id "1adb_hA9q0Qg959bl70WiRpm5a9Hdc2Ym" -O model/fastspeech2-200k.h5
RUN gdown --id "1tmmUjKIFekzlQi0-BmEcrib_QP2QbChY" -O model/mb.melgan-1000k.h5

COPY . .

CMD python3.7 app.py




