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
    model_template = argv[8]

    steps = StepsOps(SageMaker(execution_role_arn, workflow_name), workflow_name, image_uri, image_tag)
    s3_bucket_in, s3_bucket_out = (
        steps.generate_s3_bucket_name(bucket_in_template, ts),
        steps.generate_s3_bucket_name(bucket_out_template, ts),
    )
    model_path = steps.generate_s3_model_path(model_template, ts)
    model_bias_check_config, model_bias_check_processor = steps.bias_check(s3_bucket_in, s3_bucket_out, model_path)

    model_bias_check_processor.run_bias(**model_bias_check_config)

    print("Use following info as input for further steps:")
    for job in model_bias_check_processor.jobs:
        for output in job.outputs:
            print(f"Name: {output.output_name}, Path: {output.destination}")


if __name__ == "__main__":
    main(sys.argv)
