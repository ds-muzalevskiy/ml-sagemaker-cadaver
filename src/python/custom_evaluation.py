import json
import os
import pathlib
import pickle
import tarfile

from numpy import ndarray
import numpy as np
from pandas import DataFrame
import pandas as pd
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    base_path: str = "/opt/ml/processing"

    file_path: str = os.path.join(base_path, "test/test_features.csv")
    source_df: DataFrame = pd.read_csv(file_path, header=None)

    label_index: str = source_df.columns[-1]
    X: DataFrame = source_df.drop(label_index, axis=1)
    y: DataFrame = source_df[label_index]

    model_path = os.path.join(base_path, "model")

    with tarfile.open(os.path.join(model_path, "model.tar.gz")) as tar:
        tar.extractall(path=model_path)

    with open(os.path.join(model_path, "ml-model.pkl"), "rb") as inp:
        model = pickle.load(inp)
    predictions = model.predict(X)

    mse: float = mean_squared_error(y, predictions)
    std: ndarray = np.std(y - predictions)
    report_dict: dict = {
        "regression_metrics": {
            "mse": {"value": mse, "standard_deviation": std},
        },
    }

    output_dir = os.path.join(base_path, "output")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = os.path.join(output_dir, "evaluation.json")
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
