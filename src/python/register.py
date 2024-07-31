import sys

from framework.ops.base import SageMaker
from framework.ops.steps import StepsOps


def main(argv):
    workflow_name = argv[1]
    execution_role_arn = argv[2]
    image_uri = argv[3]
    image_tag = argv[4]
    ts = argv[5]
    model_path = argv[6]
    metrics_path = argv[7]
    data_bias_statistics_path = argv[8]
    data_bias_constraints_path = argv[9]

    steps = StepsOps(SageMaker(execution_role_arn, workflow_name), workflow_name, image_uri, image_tag)
    registration_config, registration_model = steps.registration(
        model_path, metrics_path, data_bias_statistics_path, data_bias_constraints_path, ts
    )

    registration_model.register(
        content_types=registration_config["content_types"],
        response_types=registration_config["response_types"],
        inference_instances=registration_config["inference_instances"],
        transform_instances=registration_config["transform_instances"],
        model_package_group_name=registration_config["model_package_group_name"],
        image_uri=registration_config["image_uri"],
        model_metrics=registration_config["model_metrics"],
        approval_status=registration_config["approval_status"],
        drift_check_baselines=registration_config["drift_check_baselines"],
    )


if __name__ == "__main__":
    main(sys.argv)
