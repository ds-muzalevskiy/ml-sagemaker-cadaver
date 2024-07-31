from sagemaker.workflow.pipeline import Pipeline


class PipelineOps:
    def build(self, pipeline_name: str, steps: [], execution_role_arn, parameters: []) -> Pipeline:
        pipeline = Pipeline(name=pipeline_name, parameters=parameters, steps=steps)
        pipeline.upsert(execution_role_arn)
        return pipeline
