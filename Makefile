.PHONY = setup-environment, format, lint, type-check \
help, deploy, ecr-login, \
build-docker, tag-docker, push-docker \
train

.DEFAULT_GOAL = help

PYTHON = poetry run python

REQUIREMENTS_FILE_NAME = requirements.txt

AWS_DEFAULT_REGION ?= eu-west-1
AWS_REGION ?= eu-west-1
AWS_ACCOUNT ?= 385715107825

ENV = dev
WORKFLOW_NAME = experimental-sm
STACK_NAME = ${WORKFLOW_NAME}

ECR_REPO = ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com
IMAGE = ${WORKFLOW_NAME}
IMAGE_URI = ${ECR_REPO}/${IMAGE}

SHORT_HASH := $(shell git rev-parse --short HEAD)
IMAGE_TAG = ${SHORT_HASH}

ROLE_NAME = ExperimentalSagemakerExecutionRole
EXECUTION_ROLE_ARN = arn:aws:iam::${AWS_ACCOUNT}:role/${ROLE_NAME}

TRAINING_RAW_S3_BUCKET_TEMPLATE = s3://${WORKFLOW_NAME}/training/
TRAINING_PREPROCESS_S3_BUCKET_TEMPLATE = s3://${WORKFLOW_NAME}/training/{}/{}/processed
TRAINING_IN_S3_BUCKET_TEMPLATE = s3://${WORKFLOW_NAME}/training/{}/{}/input
TRAINING_OUT_S3_BUCKET_TEMPLATE = s3://${WORKFLOW_NAME}/training/{}/{}
CLARIFY_OUT_S3_BUCKET_TEMPLATE = s3://${WORKFLOW_NAME}/training/{}/{}/bias
MODEL_TEMPLATE = ${TRAINING_OUT_S3_BUCKET_TEMPLATE}/output/model.tar.gz

OPS_PATH = src/python
TS := $(shell date +%s)

DRY_RUN = False

help:
	@echo "To train model type make train"

setup-environment-old:
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/901bdf0491005f1b3db41947d0d938da6838ecb9/get-poetry.py | python -; \
	poetry install

setup-environment:
	curl -sSL https://install.python-poetry.org | python3 - && \
		poetry config virtualenvs.create true && \
		poetry config virtualenvs.in-project true && \
		poetry install

format:
	@echo Formatting...
	poetry run black src/

lint:
	@echo Linting...
	poetry run flakehell lint src

type-check: lint
	@echo Type checking...
	poetry run mypy src

deploy:
	@echo "Deploying stack ${STACK_NAME}..."
	aws cloudformation deploy --stack-name "${STACK_NAME}" \
	--template-file cfn/template.yaml \
	--parameter-overrides \
	"EcrRepositoryName=${STACK_NAME}" \
	"RoleName=${ROLE_NAME}" \
	--capabilities CAPABILITY_NAMED_IAM \
	--no-fail-on-empty-changeset

create-requirements:
	@echo Exporting python depndencies...
	poetry export --without-hashes > ${REQUIREMENTS_FILE_NAME}

delete-requirements:
	@echo Removing python depndencies...
	rm -f ${REQUIREMENTS_FILE_NAME}

ecr-login:
	@echo "Ecr login..."
	aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${IMAGE_URI}

build-docker: create-requirements ecr-login
	@echo "Building docker image..."
	docker build --pull --no-cache --rm -t ${IMAGE_URI}:${IMAGE_TAG} .

tag-docker: build-docker
	@echo "Tagging docker image... "
	docker tag ${IMAGE_URI}:${IMAGE_TAG} ${IMAGE_URI}:${IMAGE_TAG}

push-docker: tag-docker
	@echo "Pushing docker image..."
	docker push ${IMAGE_URI}:${IMAGE_TAG}

ingest:
	@echo "Ingesting ..."
	${PYTHON} ${OPS_PATH}/ingestion.py ${WORKFLOW_NAME} ${EXECUTION_ROLE_ARN} ${IMAGE_URI} ${IMAGE_TAG} ${TS} \
		${TRAINING_RAW_S3_BUCKET_TEMPLATE} ${TRAINING_PREPROCESS_S3_BUCKET_TEMPLATE}

preprocess:
	@echo "Preprocessing ..."
	${PYTHON} ${OPS_PATH}/preprocess.py ${WORKFLOW_NAME} ${EXECUTION_ROLE_ARN} ${IMAGE_URI} ${IMAGE_TAG} ${TS} \
		${TRAINING_PREPROCESS_S3_BUCKET_TEMPLATE} ${TRAINING_IN_S3_BUCKET_TEMPLATE}

check-quality:
	@echo "Checking quality..."
	${PYTHON} ${OPS_PATH}/check-quality.py ${WORKFLOW_NAME} ${EXECUTION_ROLE_ARN} ${IMAGE_URI} ${IMAGE_TAG} ${TS} \
		${TRAINING_IN_S3_BUCKET_TEMPLATE} ${CLARIFY_OUT_S3_BUCKET_TEMPLATE} ${MODEL_TEMPLATE}

check-bias:
	@echo "Checking bias..."
	${PYTHON} ${OPS_PATH}/check_bias.py ${WORKFLOW_NAME} ${EXECUTION_ROLE_ARN} ${IMAGE_URI} ${IMAGE_TAG} ${TS} \
		${TRAINING_IN_S3_BUCKET_TEMPLATE} ${CLARIFY_OUT_S3_BUCKET_TEMPLATE} ${MODEL_TEMPLATE}

transform:
	@echo "Transforming (inferring) ..."
	${PYTHON} ${OPS_PATH}/transform.py ${WORKFLOW_NAME} ${EXECUTION_ROLE_ARN} ${IMAGE_URI} ${IMAGE_TAG} ${TS} \
		${TRAINING_IN_S3_BUCKET_TEMPLATE} ${TRAINING_OUT_S3_BUCKET_TEMPLATE} ${MODEL_NAME}

train:
	@echo "Training ..."
	${PYTHON} ${OPS_PATH}/train.py ${WORKFLOW_NAME} ${EXECUTION_ROLE_ARN} ${IMAGE_URI} ${IMAGE_TAG} ${TS} \
		${TRAINING_IN_S3_BUCKET_TEMPLATE} ${TRAINING_OUT_S3_BUCKET_TEMPLATE}

evaluate:
	@echo "Evaluating ..."
	${PYTHON} ${OPS_PATH}/evaluate.py ${WORKFLOW_NAME} ${EXECUTION_ROLE_ARN} ${IMAGE_URI} ${IMAGE_TAG} ${TS} \
		${TRAINING_IN_S3_BUCKET_TEMPLATE} ${MODEL_TEMPLATE}

register:
	@echo "Evaluating ..."
	${PYTHON} ${OPS_PATH}/register.py ${WORKFLOW_NAME} ${EXECUTION_ROLE_ARN} ${IMAGE_URI} ${IMAGE_TAG} ${TS} \
		${MODEL_TEMPLATE} ${METRICS_TEMPLATE} ${DATA_BIAS_STATISTICS} ${DATA_BIAS_CONSTRAINTS}

deploy-endpoint-dev deploy-endpoint-box deploy-endpoint-pro: deploy-endpoint-%:
	@echo "Deploying Endpoint on $*"
	${PYTHON} ${OPS_PATH}/deploy_endpoint.py ${WORKFLOW_NAME} ${EXECUTION_ROLE_ARN} ${IMAGE_URI} ${IMAGE_TAG} ${TS} ${MODEL_TEMPLATE} $*

test-endpoint-dev test-endpoint-box test-endpoint-pro: test-endpoint-%:
	@echo "Testing Endpoint on $*"
	${PYTHON} ${OPS_PATH}/test_endpoint.py ${WORKFLOW_NAME} ${EXECUTION_ROLE_ARN} ${IMAGE_URI} ${IMAGE_TAG} $*

remove-endpoint-dev remove-endpoint-box remove-endpoint-pro: remove-endpoint-%:
	@echo "Removing Endpoint on $*"
	${PYTHON} ${OPS_PATH}/remove_endpoint.py ${WORKFLOW_NAME} ${EXECUTION_ROLE_ARN} ${IMAGE_URI} ${IMAGE_TAG} ${TS} $*

run-pipeline-dev run-pipeline-box run-pipeline-prod: run-pipeline-%:
	@echo "Run pipeline on $*"
	${PYTHON} ${OPS_PATH}/run_pipeline.py ${WORKFLOW_NAME} ${EXECUTION_ROLE_ARN} ${IMAGE_URI} ${IMAGE_TAG} ${TS} \
		${TRAINING_RAW_S3_BUCKET_TEMPLATE} ${TRAINING_PREPROCESS_S3_BUCKET_TEMPLATE} \
		${TRAINING_IN_S3_BUCKET_TEMPLATE} ${TRAINING_OUT_S3_BUCKET_TEMPLATE} $* ${DRY_RUN}
