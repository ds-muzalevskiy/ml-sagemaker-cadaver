import sys

from framework.ops.base import SageMaker
from framework.ops.steps import StepsOps


def main(argv):
    workflow_name = argv[1]
    execution_role_arn = argv[2]
    image_uri = argv[3]
    image_tag = argv[4]
    ts = argv[5]
    s3_bucket_preprocess_template = argv[6]
    s3_bucket_in_template = argv[7]

    steps = StepsOps(SageMaker(execution_role_arn, workflow_name), workflow_name, image_uri, image_tag)
    s3_bucket_in, s3_bucket_out = (
        steps.generate_s3_bucket_name(s3_bucket_preprocess_template, ts),
        steps.generate_s3_bucket_name(s3_bucket_in_template, ts),
    )
    config, processor = steps.preprocessing(s3_bucket_in, s3_bucket_out)

    processor.run(
        code=config["code"],
        inputs=config["inputs"],
        outputs=config["outputs"],
        arguments=config["arguments"],
    )


if __name__ == "__main__":
    main(sys.argv)
