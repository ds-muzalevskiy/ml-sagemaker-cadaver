import sys

from framework.ops.base import SageMaker
from framework.ops.steps import StepsOps


def main(argv):
    workflow_name = argv[1]
    execution_role_arn = argv[2]
    image_uri = argv[3]
    image_tag = argv[4]
    ts = argv[5]
    bucket_template = argv[6]
    model_template = argv[7]

    steps = StepsOps(SageMaker(execution_role_arn, workflow_name), workflow_name, image_uri, image_tag)
    s3_bucket_in = steps.generate_s3_bucket_name(bucket_template, ts)
    model_path = steps.generate_s3_model_path(model_template, ts)

    evaluation_config, evaluation_step = steps.evaluation(s3_bucket_in, model_path)
    evaluation_step.run(
        code=evaluation_config["code"],
        inputs=evaluation_config["inputs"],
        outputs=evaluation_config["outputs"],
    )
    print("Use following info as input for further steps:")
    for job in evaluation_step.jobs:
        for output in job.outputs:
            print(f"Name: {output.output_name}, Path: {output.destination}")


if __name__ == "__main__":
    main(sys.argv)
