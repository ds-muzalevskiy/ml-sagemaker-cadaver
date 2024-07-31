import sys

from framework.ops.base import SageMaker
from framework.ops.steps import StepsOps


def main(argv):
    workflow_name = argv[1]
    execution_role_arn = argv[2]
    image_uri = argv[3]
    image_tag = argv[4]
    env = argv[5] or "dev"

    steps = StepsOps(SageMaker(execution_role_arn, workflow_name), workflow_name, image_uri, image_tag, env)
    steps.test_endpoint()


if __name__ == "__main__":
    main(sys.argv)
