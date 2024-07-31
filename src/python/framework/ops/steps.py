from typing import List, Optional

from sagemaker import FileSource, MetricsSource, clarify
import sagemaker as sm

from sagemaker.clarify import BiasConfig, DataConfig, ModelPredictedLabelConfig, SageMakerClarifyProcessor
from sagemaker.model_monitor import ModelQualityMonitor
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.estimator import Estimator
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.transformer import Transformer
from sagemaker.spark import PySparkProcessor

from framework.ops.base import SageMaker
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.clarify_check_step import DataBiasCheckConfig, ModelBiasCheckConfig
from sagemaker.workflow.quality_check_step import ModelQualityCheckConfig
from sagemaker.workflow.properties import PropertyFile

from sagemaker.model_monitor import DatasetFormat

class StepsOps:
    def __init__(self, sagemaker: SageMaker, workflow_name: str, image_uri: str, image_tag: str, env: str = "dev"):
        self.sagemaker = sagemaker
        self.workflow_name = workflow_name
        self.image_uri = image_uri
        self.image_tag = image_tag
        self.image = f"{image_uri}:{image_tag}"
        self.env = env

    def ingestion(self, s3_bucket_in: str, s3_bucket_out: str) -> (dict, ScriptProcessor):
        processor = PySparkProcessor(
            base_job_name="spark-preprocessor",
            framework_version="2.4",
            role=self.sagemaker.execution_role_arn,
            instance_count=1,
            instance_type="ml.c4.xlarge",
            max_runtime_in_seconds=1200,
        )

        config = {
            "name": "Ingestion",
            "arguments": ["--s3_input_bucket", s3_bucket_in, "--s3_output_bucket", s3_bucket_out],
            "code": "src/python/custom_ingestion.py",
        }

        return config, processor

    def preprocessing(self, s3_bucket_in: str, s3_bucket_out: str) -> (dict, ScriptProcessor):
        processor = ScriptProcessor(
            command=["python"],
            role=self.sagemaker.execution_role_arn,
            image_uri=self.image,
            instance_type="ml.c4.xlarge",
            instance_count=1,
        )

        config = {
            "name": "Preprocessing",
            "in_suffix": "processed",
            "out_suffix": "input",
            "inputs": [
                ProcessingInput(source=s3_bucket_in, destination="/opt/ml/processing/input"),
            ],
            "outputs": [
                ProcessingOutput(output_name="train_data", source="/opt/ml/processing/train"),
                ProcessingOutput(output_name="test_data", source="/opt/ml/processing/test"),
            ],
            "arguments": [
                "--train-test-split-ratio",
                "0.2",
                "--scaler-columns",
                "age,high_blood_pressure,serum_sodium",
                # "--encoder-columns",
                # "categorical_encoded,categorical_encoded_missing",
                # "--missing-categorical-values-columns",
                # "categorical_missing,categorical_encoded_missing",
                "--missing-numerical-values-columns",
                "smoking",
                "--output",
                s3_bucket_out,
            ],
            "code": "src/python/custom_preprocessing.py",
        }

        return config, processor

    def quality_check(self, s3_bucket_in: str, s3_bucket_out: str):
        model_quality_check_config = ModelQualityCheckConfig(
            baseline_dataset=s3_bucket_in,
            dataset_format=DatasetFormat.csv(header=False),
            output_s3_uri=s3_bucket_out,
            problem_type="BinaryClassification",
            inference_attribute="_c0",  # use auto-populated headers since we don't have headers in the dataset
            ground_truth_attribute="_c1",  # use auto-populated headers since we don't have headers in the dataset
        )

        check_job_config = CheckJobConfig(
            role=self.sagemaker.execution_role_arn,
            instance_count=1,
            instance_type="ml.c5.xlarge",
            volume_size_in_gb=120,
            sagemaker_session=self.sagemaker.session,
        )

        quality_check_config = {
            "model_quality_check_config": model_quality_check_config,
            "check_job_config": check_job_config,
            "model_package_group_name": self.workflow_name,
        }

        return quality_check_config, None

    def bias_check(
        self, s3_bucket_in: str, s3_bucket_out_prefix: str, model_name: Optional[str]
    ) -> (dict, ScriptProcessor):
        clarify_processor = SageMakerClarifyProcessor(
            role=self.sagemaker.execution_role_arn,
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=self.sagemaker.session,
            version="1.3-1",
        )
        suffix = "data_bias_check" if model_name else "model_bias_check"
        s3_bucket_out = f"{s3_bucket_out_prefix}/{suffix}"
        target_variable_index = 12
        data_bias_data_config = DataConfig(
            s3_data_input_path=s3_bucket_in,
            s3_output_path=s3_bucket_out,
            label=target_variable_index,
            dataset_type="text/csv",
            s3_analysis_config_output_path=f"{s3_bucket_out}/config",
        )

        data_bias_config = BiasConfig(
            label_values_or_threshold=[1], facet_name=[0, 1, 3, 10], facet_values_or_threshold=[[0], [0], [0], [1]]
        )
        data_bias_check_config = DataBiasCheckConfig(
            data_config=data_bias_data_config,
            data_bias_config=data_bias_config,
        )
        model_config = clarify.ModelConfig(
            model_name=model_name,
            instance_type="ml.m5.xlarge",
            instance_count=1,
            accept_type="text/csv",
            content_type="text/csv",
        )

        model_bias_check_config = ModelBiasCheckConfig(
            data_config=data_bias_data_config,
            data_bias_config=data_bias_config,
            model_config=model_config,
            model_predicted_label_config=ModelPredictedLabelConfig(),
        )
        check_job_config = CheckJobConfig(
            role=self.sagemaker.execution_role_arn,
            instance_count=1,
            instance_type="ml.c5.xlarge",
            volume_size_in_gb=120,
            sagemaker_session=self.sagemaker.session,
        )
        clarify_config = {
            "data_config": data_bias_data_config,
            "bias_config": data_bias_config,
            "model_config": model_config,
            "pre_training_methods": "all",
            "post_training_methods": "all",
            "model_bias_check_config": model_bias_check_config,
            "data_bias_check_config": data_bias_check_config,
            "check_job_config": check_job_config,
            "model_package_group_name": self.workflow_name,
        }

        return clarify_config, clarify_processor

    def transformation(self, s3_bucket_in: str, s3_bucket_out: str, model_name) -> (dict, Transformer):
        transformer = Transformer(
            model_name=model_name,
            instance_type="ml.m5.xlarge",
            instance_count=1,
            accept="text/csv",
            assemble_with="Line",
            output_path=f"{s3_bucket_out}/transform",
        )

        transformation_config = {
            "data": s3_bucket_in,
            "input_filter": "$[:-2]",
            "join_source": "Input",
            "output_filter": "$[-2,-1]",
            "content_type": "text/csv",
            "split_type": "Line",
        }

        return transformation_config, transformer

    def training(self, s3_bucket_in: str, s3_bucket_out: str, ts: str) -> (dict, Estimator):
        job_name = f"{self.workflow_name}-{self.image_tag}-{ts}"

        estimator = Estimator(
            self.image,
            self.sagemaker.execution_role_arn,
            1,
            "ml.c4.2xlarge",
            output_path=s3_bucket_out,
            base_job_name=job_name,
            sagemaker_session=self.sagemaker.session,
        )

        config = {
            "name": "Training",
            "job_name": job_name,
            "inputs": s3_bucket_in,
        }

        return config, estimator

    def evaluation(self, s3_bucket_in: str, model_path: str) -> (dict, Estimator):
        evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
        processor = ScriptProcessor(
            command=["python"],
            role=self.sagemaker.execution_role_arn,
            image_uri=self.image,
            instance_type="ml.c4.xlarge",
            instance_count=1,
        )

        config = {
            "name": "Evaluation",
            "inputs": [
                ProcessingInput(
                    source=model_path,
                    destination="/opt/ml/processing/model",
                ),
                ProcessingInput(
                    source=s3_bucket_in,
                    destination="/opt/ml/processing/test",
                ),
            ],
            "outputs": [
                ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/output"),
            ],
            "reports": [evaluation_report],
            "code": "src/python/custom_evaluation.py",
        }

        return config, processor

    def registration(
        self,
        model_path: str,
        model_statistics_path: str,
        data_bias_statistics: str,
        data_bias_constrains: str,
        model_bias_statistics: str,
        model_bias_constrains: str,
        bias_config_file: str,
        ts: str,
    ):
        model_name = f"{self.workflow_name}-{self.image_tag}-{ts}"
        model_group_name = self.workflow_name
        self._create_model_group_if_not_existent(model_group_name)

        model = sm.Model(
            model_data=model_path,
            image_uri=self.image,
            role=self.sagemaker.execution_role_arn,
            name=f"{model_name}-{self.env}",
            sagemaker_session=self.sagemaker.session,
        )

        metrics = sm.ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=f"{model_statistics_path}/evaluation.json",
                content_type="application/json",
            ),
            bias_pre_training=MetricsSource(
                s3_uri=data_bias_statistics,
                content_type="application/json",
            ),
            bias_post_training=MetricsSource(
                s3_uri=model_bias_statistics,
                content_type="application/json",
            ),
        )
        drift_check_baselines = DriftCheckBaselines(
            bias_pre_training_constraints=MetricsSource(
                s3_uri=data_bias_constrains,
                content_type="application/json",
            ),
            bias_config_file=FileSource(
                s3_uri=bias_config_file,
                content_type="application/json",
            ),
            bias_post_training_constraints=MetricsSource(
                s3_uri=model_bias_constrains,
                content_type="application/json",
            ),
        )
        config = {
            "name": "Registration",
            "content_types": ["text/csv", "application/json"],
            "response_types": ["text/csv", "application/json"],
            "inference_instances": ["ml.t2.medium", "ml.m5.xlarge"],
            "transform_instances": ["ml.m5.xlarge"],
            "image_uri": self.image,
            "approval_status": "Approved",
            "model_metrics": metrics,
            "model_package_group_name": model_group_name,
            "drift_check_baselines": drift_check_baselines,
            "model": model,
        }

        return config, model

    def deployment(self, s3_bucket_in: str, _, ts: str):
        model_name = f"{self.workflow_name}-{self.image_tag}-{ts}"

        # endpoints_names = self._get_endpoints_names(self.sagemaker.client)
        # update_endpoint_ = self._is_endpoint_in_endpoints_names(f"{model_name}-{self.env}", endpoints_names)
        # if update_endpoint_:
        #     print("Updating Existing Endpoint: ", model_name)
        #     self.remove_endpoint()
        # else:
        #     print("Creating New Endpoint: ", model_name)

        model = sm.Model(
            model_data=s3_bucket_in,
            image_uri=self.image,
            role=self.sagemaker.execution_role_arn,
            name=f"{model_name}-{self.env}",
            sagemaker_session=self.sagemaker.session,
        )

        config = {
            "name": "Deploying",
            "model_name": model_name,
            "initial_instance_count": 1,
            "instance_type": "ml.t2.medium",
        }

        return config, model

    def test_endpoint(self):
        endpoint_name = f"{self.workflow_name}-{self.env}"
        test_payload = "-0.84,1.31,-1.43,0.0,0,1380,0,25,271000,0.9,1,38"

        response = self.sagemaker.runtime_client.invoke_endpoint(
            EndpointName=endpoint_name, ContentType="text/csv", Body=test_payload
        )

        status_code = response["ResponseMetadata"]["HTTPStatusCode"]
        body = response["Body"].read()

        is_status_code_ok = status_code == 200
        is_classified_1 = body == b"1\n"

        assert is_status_code_ok & is_classified_1

    def remove_endpoint(self, ts: str):
        endpoint_name = f"{self.workflow_name}-{self.image_tag}-{ts}-{self.env}"
        cleanup_endpoint_response = self.sagemaker.client.delete_endpoint(EndpointName=endpoint_name)
        if cleanup_endpoint_response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            print(f"Cleaned up {self.env} Endpoint.")
        cleanup_config_response = self.sagemaker.client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        if cleanup_config_response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            print("Cleaned up Dev Endpoint Config.")

    def _get_endpoints_names(self, sm_client):
        list_endpoints_response = sm_client.list_endpoints(
            SortBy="Name", NameContains=self.workflow_name, StatusEquals="InService", MaxResults=100
        )
        endpoints = list_endpoints_response["Endpoints"]
        endpoints_names = [endpoint["EndpointName"] for endpoint in endpoints]
        return endpoints_names

    def _is_endpoint_in_endpoints_names(self, endpoint_name: str, endpoints_names: List[str]) -> bool:
        is_existing_endpoint = False
        if endpoint_name in endpoints_names:
            is_existing_endpoint = True

        return is_existing_endpoint

    def generate_s3_bucket_name(self, s3_bucket_template: str, ts: str) -> str:
        return s3_bucket_template.format(self.image_tag, ts)

    def generate_s3_model_path(self, model_template: str, ts: str) -> str:
        model_name = f"{self.workflow_name}-{self.image_tag}-{ts}"
        return model_template.format(f"{self.image_tag}/{ts}", model_name)

    def _create_model_group_if_not_existent(self, model_group_name: str) -> None:
        model_groups = self.sagemaker.client.list_model_package_groups()
        model_group_names = [group["ModelPackageGroupName"] for group in model_groups["ModelPackageGroupSummaryList"]]
        if model_group_name not in model_group_names:
            model_package_group_input_dict = {
                "ModelPackageGroupName": model_group_name,
                "ModelPackageGroupDescription": f"Model package group for {model_group_name}",
            }
            self.sagemaker.client.create_model_package_group(**model_package_group_input_dict)
