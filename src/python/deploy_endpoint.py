import sys

from framework.ops.base import SageMaker
from framework.ops.steps import StepsOps


def main(argv):
    workflow_name = argv[1]
    execution_role_arn = argv[2]
    image_uri = argv[3]
    image_tag = argv[4]
    ts = argv[5]
    model_template = argv[6]
    env = argv[7] or "dev"

    steps = StepsOps(SageMaker(execution_role_arn, workflow_name), workflow_name, image_uri, image_tag, env)
    s3_bucket_in = steps.generate_s3_model_path(model_template, ts)
    deployment_config, deployment_model = steps.deployment(s3_bucket_in, None, ts)

    deployment_model.deploy(
        initial_instance_count=1,
        instance_type=deployment_config["instance_type"],
        endpoint_name=f'{deployment_config["model_name"]}-{env}',
    )


if __name__ == "__main__":
    main(sys.argv)
