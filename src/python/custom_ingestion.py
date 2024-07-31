from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

from pyspark import sql
from pyspark.sql import SparkSession


def main():
    parser = argparse.ArgumentParser(description="app inputs and outputs")
    parser.add_argument("--s3_input_bucket", type=str, help="s3 input bucket")
    parser.add_argument("--s3_output_bucket", type=str, help="s3 output bucket")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("PySparkApp").getOrCreate()

    input_path: str = os.path.join(args.s3_input_bucket, "heart_failure_clinical_records_dataset.csv")
    total_df: sql.DataFrame = spark.read.csv(input_path, header=True)

    ###
    # Write your custom ingestion code here:
    ###

    total_df.write.mode("overwrite").parquet(args.s3_output_bucket)


if __name__ == "__main__":
    main()
