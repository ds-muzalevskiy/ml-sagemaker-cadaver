from __future__ import print_function

import io
import os
import pickle
from typing import Optional

import flask
from numpy import ndarray
from pandas import DataFrame, read_csv, read_json
from sklearn.ensemble import VotingClassifier


prefix = "/opt/ml/"


class ScoringService(object):
    model = None               

    @classmethod
    def get_model(cls) -> VotingClassifier:
        if cls.model is None:
            model_path = os.path.join(prefix, "model")
            with open(os.path.join(model_path, "ml-model.pkl"), "rb") as inp:
                cls.model = pickle.load(inp)
        return cls.model

    @classmethod
    def predict(cls, input) -> ndarray:
        clf = cls.get_model()
        return clf.predict(input)


app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    health = ScoringService.get_model() is not None  

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    content_type: str = flask.request.content_type
    data: Optional[DataFrame] = read_data(flask.request.data, content_type)

    if data is None:
        return flask.Response(response="This predictor only supports CSV or JSON data", status=415, mimetype="text/plain")

    print("Invoked with {} records".format(data.shape[0]))
    predictions: ndarray = ScoringService.predict(data)

    result: Optional[str] = write_response(predictions, content_type)

    return flask.Response(response=result, status=200, mimetype=content_type)


def read_data(request_data: bytes, content_type: str) -> Optional[DataFrame]:
    if content_type == "text/csv":
        return read_csv(io.BytesIO(request_data), header=None)
    elif content_type == "application/json":
        df = read_json(io.BytesIO(request_data), lines=True)

        # The feature names should match those that were passed during fit.
        # Starting sklearn version 1.2, an error will be raised.
        feature_names = ScoringService.get_model().feature_names_in_
        df = df[feature_names]
        return df
    else:
        return None


def write_response(predictions: ndarray, content_type: str) -> Optional[str]:
    out = io.StringIO()
    if content_type == "text/csv":
        DataFrame({"prediction": predictions}).to_csv(out, header=False, index=False)
    elif content_type == "application/json":
        DataFrame({"prediction": predictions}).to_json(out, orient="records")
    result = out.getvalue()
    return result


if __name__ == "__main__":
    app.run(debug=True, use_debugger=False, use_reloader=False)

