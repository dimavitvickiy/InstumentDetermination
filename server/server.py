import os
import pickle

from flask import Flask, render_template, request
from wtforms import Form, fields

import instrument_data
from feature_extraction.feature_extractor import extract_features

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
            with open('model.pickle', 'rb') as f:
                classifier = pickle.load(f)

            features = extract_features(os.path.join(UPLOAD_PATH, audiofile.filename))

            predict_x = {str(feature_col): [feature] for feature_col, feature in
                         zip(range(instrument_data.FEATURES_NUMBER), features)}
            predictions = classifier.predict(
                input_fn=lambda: instrument_data.eval_input_fn(
                    predict_x,
                    labels=None,
                    batch_size=50))

            for pred_dict in predictions:
                class_id = pred_dict['class_ids'][0]
                probability = pred_dict['probabilities'][class_id]

                return render_template(
                    "main.html.jinja2",
                    form=form,
                    instrument={
                        'name': instrument_data.INSTRUMENTS[class_id],
                        'probability': 100 * probability,
                    }
                )
    return render_template("main.html.jinja2", form=form, instrument={})


if __name__ == '__main__':
    app.run(debug=True)
