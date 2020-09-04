FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
RUN apt-get update
RUN apt-get install -y python3.7-dev python3-pip wget libsndfile1

RUN mkdir /workspace
WORKDIR /workspace

COPY requirements.txt .

RUN python3.7 -m pip install -U pip setuptools
RUN python3.7 -m pip install -r requirements.txt

RUN mkdir ./model

RUN gdown --id "1n36vcrEPQ0SyL7wVrYsNVrPiuGhiOF4h" -O model/tacotron2-100k.h5
RUN gdown --id "1PAFpsILkih5zSTbYQw-hpAe5eaNJh0hb" -O model/fastspeech2-200k.h5
RUN gdown --id "17Db2R2k8mJmbO5utX80AVK8ssAr-JpCB" -O model/mb.melgan-920k.h5

RUN mkdir ./data

COPY . .

CMD python3.7 app.py




