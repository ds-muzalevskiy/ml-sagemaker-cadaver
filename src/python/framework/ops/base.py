from dataclasses import dataclass, field

import boto3
import sagemaker as sm

from botocore.client import BaseClient


@dataclass
class SageMaker:
    execution_role_arn: str
    bucket_name: str
    session: sm.Session = field(init=False)
    client: BaseClient = field(default=boto3.client("sagemaker"))
    runtime_client: BaseClient = field(default=boto3.client("runtime.sagemaker"))

    def __post_init__(self):
        self.session = sm.Session(default_bucket=self.bucket_name)
