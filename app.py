import os
import io
import uuid
import shutil

import threading
import time
from queue import Empty, Queue

import tensorflow as tf

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import AutoProcessor

from scipy.io import wavfile

import nltk
nltk.download('punkt')

DATA_FOLDER = 'data'

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

requests_queue = Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1
########################################################################################
processor = AutoProcessor.from_pretrained(pretrained_path="mapper/ljspeech_mapper.json")

tacotron2_config = AutoConfig.from_pretrained('conf/tacotron2.v1.yaml')
tacotron2 = TFAutoModel.from_pretrained(
    config=tacotron2_config,
    pretrained_path="model/tacotron2-120k.h5",
    training=False, 
    name="tacotron2"
)

fastspeech_config = AutoConfig.from_pretrained('conf/fastspeech.v1.yaml')
fastspeech = TFAutoModel.from_pretrained(
    config=fastspeech_config,
    pretrained_path="model/fastspeech-150k.h5",
    name="fastspeech"
)

fastspeech2_config = AutoConfig.from_pretrained('conf/fastspeech2.v1.yaml')
fastspeech2 = TFAutoModel.from_pretrained(
    config=fastspeech2_config,
    pretrained_path="model/fastspeech2-150k.h5",
    name="fastspeech2"
)

melgan_config = AutoConfig.from_pretrained('conf/melgan.v1.yaml')
melgan = TFAutoModel.from_pretrained(
    config=melgan_config,
    pretrained_path="model/melgan-1M6.h5",
    name="melgan"
)

melgan_stft_config = AutoConfig.from_pretrained('conf/melgan.stft.v1.yaml')
melgan_stft = TFAutoModel.from_pretrained(
    config=melgan_stft_config,
    pretrained_path="model/melgan.stft-2M.h5",
    name="melgan_stft"
)

mb_melgan_config = AutoConfig.from_pretrained('conf/multiband_melgan.v1.yaml')
mb_melgan = TFAutoModel.from_pretrained(
    config=mb_melgan_config,
    pretrained_path="model/mb.melgan-940k.h5",
    name="mb_melgan"
)

model_name = {
    "TACOTRON": tacotron2,
    "FASTSPEECH": fastspeech,
    "FASTSPEECH2": fastspeech2,
    "MELGAN": melgan,
    "MELGAN-STFT": melgan_stft,
    "MB-MELGAN": mb_melgan
}
########################################################################################
def do_synthesis(input_text, text2mel_model, vocoder_model, text2mel_name, vocoder_name):
    input_ids = processor.text_to_sequence(input_text)

    # text2mel part
    if text2mel_name == "TACOTRON":
        _, mel_outputs, stop_token_prediction, alignment_history = text2mel_model.inference(
            tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            tf.convert_to_tensor([len(input_ids)], tf.int32),
            tf.convert_to_tensor([0], dtype=tf.int32)
    )
    elif text2mel_name == "FASTSPEECH":
        mel_before, mel_outputs, duration_outputs = text2mel_model.inference(
            input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
            speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    )
    elif text2mel_name == "FASTSPEECH2":
        mel_before, mel_outputs, duration_outputs, _, _ = text2mel_model.inference(
            tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
            speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
            f0_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
            energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    )
    else:
        raise ValueError("Only TACOTRON, FASTSPEECH, FASTSPEECH2 are supported on text2mel_name")

    # vocoder part
    if vocoder_name == "MELGAN" or vocoder_name == "MELGAN-STFT":
        audio = vocoder_model(mel_outputs)[0, :, 0]
    elif vocoder_name == "MB-MELGAN":
        audio = vocoder_model(mel_outputs)[0, :, 0]
    else:
        raise ValueError("Only MELGAN, MELGAN-STFT and MB_MELGAN are supported on vocoder_name")

    return audio.numpy()

def run(input_text, feature_generator, vocoder):
    try:
        audios = do_synthesis(input_text, model_name[feature_generator], model_name[vocoder], feature_generator, vocoder)

        f_id = str(uuid.uuid4())

        os.makedirs(os.path.join(DATA_FOLDER, f_id), exist_ok=True)

        f_path = os.path.join(DATA_FOLDER, f_id, "result.wav")

        wavfile.write(f_path, 22050, audios)

        with open(f_path, 'rb') as wav:
            wav_bytes = wav.read()

        wav_io = io.BytesIO(wav_bytes)
        wav_io.seek(0)

        shutil.rmtree(os.path.join(DATA_FOLDER, f_id))

        return wav_io

    except Exception as e:
        print(e)
        return 400

def handle_requests_by_batch():
    try:
        while True:
            requests_batch = []

            while not (
              len(requests_batch) >= BATCH_SIZE # or
              #(len(requests_batch) > 0 #and time.time() - requests_batch[0]['time'] > BATCH_TIMEOUT)
            ):
              try:
                requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
              except Empty:
                continue

            batch_outputs = []

            for request in requests_batch:
                batch_outputs.append(run(request['input'][0], request['input'][1], request['input'][2]))

            for request, output in zip(requests_batch, batch_outputs):
                request['output'] = output

    except Exception as e:
        while not requests_queue.empty():
            requests_queue.get()
        print(e)

threading.Thread(target=handle_requests_by_batch).start()
#######################################################################################
app = Flask(__name__)
cors = CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print(requests_queue.qsize())

        input_text = request.form["input_text"]
        feature_generator = request.form["feature_generator"]
        vocoder = request.form["vocoder"]

        if feature_generator not in model_name:
            return jsonify({"error": "Only TACOTRON, FASTSPEECH, FASTSPEECH2 are supported on text2mel_name"}), 400

        if vocoder not in model_name:
            return jsonify({"error": "Only MELGAN, MELGAN-STFT and MB_MELGAN are supported on vocoder_name"}), 400

        if requests_queue.qsize() >= 1:
            return jsonify({"error": "Too Many Requests"}), 429

        req = {
            'input': [input_text, feature_generator, vocoder]
        }

        requests_queue.put(req)

        while 'output' not in req:
            time.sleep(CHECK_INTERVAL)

        result = req['output']
        print(type(result))

        if req['output'] == 400:
            return jsonify({"error": "Generate TTS error! check your input data"}), 400

        return send_file(result, mimetype="audio/wav")

    except Exception as e:
        print(e)
        return jsonify({"error": "TensorFlowTTS-english server error"}), 500

@app.route('/health')
def health():
    return "ok"


@app.route('/')
def main():
    return "TensorFlowTTS-english"


if __name__ == "__main__":
    from waitress import serve
    serve(app, host='0.0.0.0', port=80)
