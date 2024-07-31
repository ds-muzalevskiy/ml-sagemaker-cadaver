import argparse
import os

from numpy.typing import ArrayLike
from typing import List

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer


def get_column_list(columns: str) -> List[str]:
    return columns.split(",") if columns else []


def to_dataframe(x: ArrayLike, transformer: ColumnTransformer) -> pd.DataFrame:
    return pd.DataFrame(
        x,
        columns=remove_prefix_from_feature_list(transformer),
    )


def remove_prefix_from_feature_list(transformer: ColumnTransformer) -> List[str]:
    prefixes: List[str] = [transformer[0] for transformer in transformer.transformers] + ["remainder"]
    column_names: List[str] = []
    for column in transformer.get_feature_names_out():
        for prefix in prefixes:
            column = column.replace(f"{prefix}__", "")
        column_names.append(column)
    return column_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="/opt/ml/processing/output")
    parser.add_argument("--train-test-split-ratio", type=float, default=0.3)
    parser.add_argument("--scaler-columns", type=str, default="")
    parser.add_argument("--encoder-columns", type=str, default="")
    parser.add_argument("--missing-categorical-values-columns", type=str, default="")
    parser.add_argument("--missing-numerical-values-columns", type=str, default="")

    args, _ = parser.parse_known_args()
    print("Received arguments {}".format(args))
    base_dir = "/opt/ml/processing"
    input_path = os.path.join(base_dir, "input")
    paths = [os.path.join(input_path, path) for path in os.listdir(input_path) if path.endswith(".parquet")]
    df = pd.concat([pd.read_parquet(file_path) for file_path in paths])

    train_set, test_set = train_test_split(
        df,
        test_size=args.train_test_split_ratio,
        random_state=0,
    )

    missing_categorical_values_columns = get_column_list(args.missing_categorical_values_columns)
    encoder_columns = get_column_list(args.encoder_columns)
    missing_numerical_values_columns = get_column_list(args.missing_numerical_values_columns)
    scaler_columns = get_column_list(args.scaler_columns)

    SimpleImputer.get_feature_names_out = lambda self, names=None: self.feature_names_in_
    OrdinalEncoder.get_feature_names_out = lambda self, names=None: self.feature_names_in_

    missing_values_cat_transformer = make_column_transformer(
        (
            SimpleImputer(missing_values=np.nan, strategy="most_frequent"),
            missing_categorical_values_columns,
        ),
        (
            SimpleImputer(missing_values=np.nan, strategy="mean"),
            missing_numerical_values_columns,
        ),
        remainder="passthrough",
    )

    encoder_transformer = make_column_transformer(
        (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), encoder_columns),
        remainder="passthrough",
    )
    scaler_transformer = make_column_transformer(
        (StandardScaler(), scaler_columns),
        remainder="passthrough",
    )

    pipeline = make_pipeline(
        missing_values_cat_transformer,
        (FunctionTransformer(to_dataframe, kw_args={"transformer": missing_values_cat_transformer})),
        encoder_transformer,
        (FunctionTransformer(to_dataframe, kw_args={"transformer": encoder_transformer})),
        scaler_transformer,
        (FunctionTransformer(to_dataframe, kw_args={"transformer": scaler_transformer})),
    )

    print("Running preprocessing and feature engineering transformations")
    train_features = pipeline.fit_transform(train_set)
    test_features = pipeline.transform(test_set)

    train_features_output_path = os.path.join(f"{args.output}/train_features.csv")
    test_features_output_path = os.path.join(f"{args.output}/test_features.csv")

    pd.DataFrame(train_features).to_csv(train_features_output_path, header=False, index=False)
    pd.DataFrame(test_features).to_csv(test_features_output_path, header=False, index=False)

    # To ensure smooth data lineage in the pipelines,
    # We have to repeat writing the paths defined as output
    pd.DataFrame(train_features).to_csv(f"{base_dir}/train/train_features.csv", header=False, index=False)
    pd.DataFrame(test_features).to_csv(f"{base_dir}/test/test_features.csv", header=False, index=False)
