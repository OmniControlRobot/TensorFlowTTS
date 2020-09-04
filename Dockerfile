FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
RUN apt-get update
RUN apt-get install -y python3.7-dev python3-pip wget libsndfile1

RUN mkdir /workspace
WORKDIR /workspace

COPY requirements.txt .

RUN python3.7 -m pip install -U pip setuptools
RUN python3.7 -m pip install -r requirements.txt

RUN mkdir ./model

RUN gdown --id "12jvEO1VqFo1ocrgY9GUHF_kVcLn3QaGW" -O model/tacotron2-120k.h5
RUN gdown --id "1T5GOE_M27zJlCAjnanpOS9HBPUcdE9sB" -O model/fastspeech-150k.h5
RUN gdown --id "1EhMD20uAFlKsii1lMnlkrsenVTFKM0ld" -O model/fastspeech2-150k.h5

RUN gdown --id "1A3zJwzlXEpu_jHeatlMdyPGjn1V7-9iG" -O model/melgan-1M6.h5
RUN gdown --id "1WB5iQbk9qB-Y-wO8BU6S2TnRiu4VU5ys" -O model/melgan.stft-2M.h5
RUN gdown --id "1kChFaLI7slrTtuk3pvcOiJwJDCygsw9C" -O model/mb.melgan-940k.h5

RUN mkdir ./data

COPY . .

CMD python3.7 app.py




