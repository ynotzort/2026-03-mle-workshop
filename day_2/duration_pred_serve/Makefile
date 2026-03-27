MODEL_PATH=models/2022-01.bin
export MODEL_PATH

MODEL_VERSION=model_v1_2022_01
export MODEL_VERSION

.PHONY: serve
serve:
	uv run fastapi dev src/duration_pred_serve/serve.py
	
.PHONY: predict-test
predict-test:
	uv run python predict-test.py
	
docker_build:
	docker build -t duration-prediction:latest .
	
docker_run: docker_build
	docker run -it -p 8080:8080 duration-prediction:latest