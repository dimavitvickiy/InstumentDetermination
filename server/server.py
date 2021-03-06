import os
import pickle
from collections import OrderedDict

from flask import Flask, render_template, request
from wtforms import Form, fields
import numpy as np

import instrument_data
from extract_all_features_cnn import DURATION
from feature_extraction.feature_extractor import extract_features
from feature_extraction.feature_extractor_cnn import extract_features as extract_features_cnn

app = Flask(__name__)
UPLOAD_PATH = 'files'


class AudioForm(Form):
    audio_file = fields.FileField('Audiofile')


@app.route("/", methods=['GET', 'POST'])
def classification():
    form = AudioForm(request.form)
    if request.method == 'POST':
        audiofile = request.files.get(form.audio_file.name)
        if audiofile:
            audiofile.save(
                os.path.join(UPLOAD_PATH, request.files[form.audio_file.name].filename),
            )
            with open('model_mlp.pickle', 'rb') as f:
                classifier = pickle.load(f)

            with open('model_cnn.pickle', 'rb') as f:
                classifier_cnn = pickle.load(f)

            features = extract_features(os.path.join(UPLOAD_PATH, audiofile.filename))
            features_cnn = extract_features_cnn(os.path.join(UPLOAD_PATH, audiofile.filename))[:, :DURATION]
            predict_x = OrderedDict()
            for feature_col, feature in zip(range(instrument_data.FEATURES_NUMBER), features):
                predict_x[str(feature_col)] = [feature]

            predictions = classifier.predict(
                input_fn=lambda: instrument_data.eval_input_fn(
                    predict_x,
                    labels=None,
                    batch_size=10))

            features_cnn = features_cnn.reshape([1] + list(features_cnn.shape[:]) + [-1])
            predictions_cnn = classifier_cnn.predict(features_cnn)

            for pred_dict in predictions:
                class_id = pred_dict['class_ids'][0]
                probability = pred_dict['probabilities'][class_id]

            class_id_cnn = np.where(predictions_cnn[0] == max(predictions_cnn[0]))[0][0]
            probability_cnn = predictions_cnn[0][class_id_cnn]

            return render_template(
                "main.html.jinja2",
                form=form,
                instrument={
                    'name': instrument_data.INSTRUMENTS_UKR[class_id],
                    'probability': 100 * probability,
                    'probability_cnn': 100 * probability_cnn,
                    'name_cnn': instrument_data.INSTRUMENTS_UKR[class_id_cnn],
                }
            )
    return render_template("main.html.jinja2", form=form, instrument={})


if __name__ == '__main__':
    app.run(debug=True)
