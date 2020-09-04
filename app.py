import requests
import io
import json

from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS

########################################################################################
URL_PATH = 'url_tts.json'

with open(URL_PATH) as f:
    url_dict = json.load(f)
########################################################################################


def run(language, input_text, feature_generator, vocoder):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        payload = {'input_text': input_text, 'feature_generator': feature_generator, 'vocoder': vocoder}

        url = url_dict[language]

        session = requests.Session()
        response = session.post(url, headers=headers, data=payload)

        return response

    except Exception as e:
        print(e)
        return 500
########################################################################################


app = Flask(__name__, template_folder='templates')
cors = CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        language = request.form["language"]
        input_text = request.form["input_text"]
        feature_generator = request.form["feature_generator"]
        vocoder = request.form["vocoder"]

        response = run(language, input_text, feature_generator, vocoder)

        if response.status_code != 200:
            print(response.status_code)

            if response.status_code == 429:
                return jsonify({'error': 'Too much request'}), 429
            elif response.status_code == 500:
                return jsonify({'error': 'Generate TTS error! input text is too long'}), 500

        result = response.content

        byte_io = io.BytesIO(result)
        byte_io.seek(0)

        return send_file(byte_io, mimetype="audio/wav")

    except Exception as e:
        print(e)
        return "check your input data", 400


@app.route('/health')
def health():
    return "ok"


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == "__main__":
    from waitress import serve
    serve(app, host='0.0.0.0', port=80)
