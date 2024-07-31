import sys

from sagemaker.processing import ProcessingInput
from sagemaker.workflow.quality_check_step import QualityCheckStep
from sagemaker.workflow.clarify_check_step import ClarifyCheckStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.parameters import ParameterFloat
from sagemaker.workflow.step_collections import RegisterModel

from framework.ops.base import SageMaker
from framework.ops.steps import StepsOps
from framework.ops.pipeline import PipelineOps

from sagemaker.inputs import CreateModelInput, TransformInput
from sagemaker.workflow.steps import CreateModelStep, ProcessingStep, TrainingStep, TransformStep

import json
from pprint import pprint


def main(argv):
    workflow_name = argv[1]
    execution_role_arn = argv[2]
    image_uri = argv[3]
    image_tag = argv[4]
    ts = argv[5]
    s3_bucket_raw_template = argv[6]
    s3_bucket_preprocess_template = argv[7]
    s3_bucket_in_template = argv[8]
    s3_bucket_out_template = argv[9]
    env = argv[10] or "dev"
    dry_run = argv[11] == "True"

    mse_threshold = ParameterFloat(name="MseThreshold", default_value=0.4)
    steps = StepsOps(SageMaker(execution_role_arn, workflow_name), workflow_name, image_uri, image_tag, env)
    s3_bucket_in, s3_bucket_out = (
        steps.generate_s3_bucket_name(s3_bucket_raw_template, ts),
        steps.generate_s3_bucket_name(s3_bucket_preprocess_template, ts),
    )
    ingestion_config, ingestion_processor = steps.ingestion(s3_bucket_in, s3_bucket_out)

    ingestion_step = ProcessingStep(
        name=ingestion_config["name"],
        processor=ingestion_processor,
        job_arguments=ingestion_config["arguments"],
        code=ingestion_config["code"],
    )

    s3_bucket_in, s3_bucket_out = (
        steps.generate_s3_bucket_name(s3_bucket_preprocess_template, ts),
        steps.generate_s3_bucket_name(s3_bucket_in_template, ts),
    )
    preprocessing_config, preprocessing_processor = steps.preprocessing(s3_bucket_in, s3_bucket_out)

    preprocessing_step = ProcessingStep(
        name=preprocessing_config["name"],
        inputs=preprocessing_config["inputs"],
        outputs=preprocessing_config["outputs"],
        processor=preprocessing_processor,
        job_arguments=preprocessing_config["arguments"],
        code=preprocessing_config["code"],
    )

    s3_bucket_base_path = steps.generate_s3_bucket_name(s3_bucket_out_template, ts)

    s3_bucket_in = preprocessing_step.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri
    data_bias_config, data_bias_processor = steps.bias_check(s3_bucket_in, s3_bucket_base_path, None)

    data_bias_check_step = ClarifyCheckStep(
        name="DataBiasCheckStep",
        clarify_check_config=data_bias_config["data_bias_check_config"],
        check_job_config=data_bias_config["check_job_config"],
        skip_check=True,
        register_new_baseline=True,
        # supplied_baseline_constraints=supplied_baseline_constraints_data_bias,
        model_package_group_name=data_bias_config["model_package_group_name"],
    )
    s3_bucket_in, s3_bucket_out = (
        steps.generate_s3_bucket_name(s3_bucket_in_template, ts),
        steps.generate_s3_bucket_name(s3_bucket_out_template, ts),
    )
    training_config, training_estimator = steps.training(s3_bucket_in, s3_bucket_out, ts)

    training_step = TrainingStep(
        name=training_config["name"],
        estimator=training_estimator,
        inputs=training_config["inputs"],
    )

    s3_bucket_in = training_step.properties.ModelArtifacts.S3ModelArtifacts
    deployment_config, deployment_model = steps.deployment(s3_bucket_in, None, ts)

    deployment_step = CreateModelStep(
        name=deployment_config["name"],
        model=deployment_model,
        inputs=CreateModelInput(
            instance_type=deployment_config["instance_type"],
        ),
    )
    s3_bucket_in = preprocessing_step.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri
    model_name = deployment_step.properties.ModelName

    transformation_config, transformer = steps.transformation(
        preprocessing_step.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri,
        s3_bucket_out, model_name
    )
    transform_step = TransformStep(
        name="Transform",
        transformer=transformer,
        inputs=TransformInput(
            **transformation_config
        ),
    )

    model_quality_check_config, _ = steps.quality_check(
        transform_step.properties.TransformOutput.S3OutputPath,
        s3_bucket_out)
    model_quality_check_step = QualityCheckStep(
        name="ModelQualityCheckStep",
        skip_check=True,
        register_new_baseline=True,
        quality_check_config=model_quality_check_config["model_quality_check_config"],
        check_job_config=model_quality_check_config["check_job_config"],
        # supplied_baseline_statistics=quality_check_config["supplied_baseline_statistics_model_quality"],
        model_package_group_name=model_quality_check_config["model_package_group_name"],
    )

    model_bias_check_config, _ = steps.bias_check(s3_bucket_in, s3_bucket_base_path, model_name)
    model_bias_check_step = ClarifyCheckStep(
        name="ModelBiasCheckStep",
        skip_check=True,
        register_new_baseline=True,
        clarify_check_config=model_bias_check_config["model_bias_check_config"],
        check_job_config=model_bias_check_config["check_job_config"],
        # supplied_baseline_constraints=model_bias_config["supplied_baseline_constraints_model_bias"],
        model_package_group_name=model_bias_check_config['model_package_group_name'],
    )

    evaluation_config, evaluation_processing = steps.evaluation(s3_bucket_in, ts)
    evaluation_step = ProcessingStep(
        name=evaluation_config["name"],
        processor=evaluation_processing,
        # We cannot reuse the inputs from the evaluation config,
        # since e.g. the model path differs depending on the execution strategy.
        # With estimator.fit, the model path is /WORKFLOW_NAME-IMAGE_TAG-TS/output/model.tar.gz
        # With the pipeline, the naming always consists of /pipelines-$(hash)-Training-$hash/output/model.tar.gz
        # Therefore, we use the output of the preprocessing/training step as input for evaluation
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model"
            ),
            ProcessingInput(
                source=preprocessing_step.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=evaluation_config["outputs"],
        code=evaluation_config["code"],
        property_files=evaluation_config["reports"],
    )

    evaluation_fail_step = FailStep(
        name="EvaluationFailed",
        error_message=Join(on=" ", values=["Execution failed due to MSE >", mse_threshold]),
    )
    registration_config, model = steps.registration(
        training_step.properties.ModelArtifacts.S3ModelArtifacts,
        evaluation_step.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"],
        data_bias_check_step.properties.CalculatedBaselineConstraints,
        data_bias_check_step.properties.BaselineUsedForDriftCheckConstraints,
        model_bias_check_step.properties.CalculatedBaselineConstraints,
        model_bias_check_step.properties.BaselineUsedForDriftCheckConstraints,
        model_bias_check_config["model_bias_check_config"].monitoring_analysis_config_uri,
        ts,
    )
    registration_step = RegisterModel(**registration_config)

    condition_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(  # type: ignore
            step_name=evaluation_step.name,
            property_file=evaluation_config["reports"][0],
            json_path="regression_metrics.mse.value",
        ),
        right=mse_threshold,
    )

    condition_step = ConditionStep(
        name="DeploymentThresholdCondition",
        conditions=[condition_lte],
        if_steps=[registration_step],
        else_steps=[evaluation_fail_step],
    )

    preprocessing_step.add_depends_on([ingestion_step])
    training_step.add_depends_on([preprocessing_step])
    evaluation_step.add_depends_on([training_step])

    pipeline = PipelineOps().build(
        steps.workflow_name,
        [
            ingestion_step,
            preprocessing_step,
            data_bias_check_step,
            training_step,
            deployment_step,
            evaluation_step,
            transform_step,
            model_quality_check_step,
            model_bias_check_step,
            condition_step,
        ],
        execution_role_arn,
        [mse_threshold],
    )

    if dry_run:
        definition = json.loads(pipeline.definition())
        pprint(definition)
    else:
        # pipeline.update(execution_role_arn)
        execution = pipeline.start()
        execution.describe()
        execution.wait()


if __name__ == "__main__":
    main(sys.argv)
