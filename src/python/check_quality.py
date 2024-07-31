import sys

from framework.ops.base import SageMaker
from framework.ops.steps import StepsOps


def main(argv):
    workflow_name = argv[1]
    execution_role_arn = argv[2]
    image_uri = argv[3]
    image_tag = argv[4]
    ts = argv[5]
    bucket_in_template = argv[6]
    bucket_out_template = argv[7]

    steps = StepsOps(SageMaker(execution_role_arn, workflow_name), workflow_name, image_uri, image_tag)
    s3_bucket_in, s3_bucket_out = (
        steps.generate_s3_bucket_name(bucket_in_template, ts),
        steps.generate_s3_bucket_name(bucket_out_template, ts),
    )
    model_quality_check_config, _ = steps.quality_check(s3_bucket_in, s3_bucket_out)


if __name__ == "__main__":
    main(sys.argv)
