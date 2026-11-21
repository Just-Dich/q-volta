# Docker

> Status: the Dockerfile and compose file were written on a machine **without a container runtime**, so the image has not been built or
> run by the authors yet. The build commands in it are the ones used for the native build (`cmake -DCMAKE_BUILD_TYPE=Release`, targets
> `qv_server qv_run qv_tok`), and the base-image tags (`nvidia/cuda:12.9.1-devel|runtime-ubuntu24.04`) were checked to exist on Docker Hub.
> Treat it as a starting point and report problems.

Weights are not baked into the image. Mount a directory with the model and tokenizer at `/models`.

```bash
# get the weights (single-file .qv) and tokenizer
hf download q-project/Qwen3.8-27B-QVolta qwen38-27b-e1.qv tokenizer.json --local-dir models
mv models/qwen38-27b-e1.qv models/model.qv

# GPU image (Tesla V100 / sm_70 kernels; needs the NVIDIA container toolkit)
docker build -t q-volta .
docker run --rm --gpus all -p 8080:8080 -v $PWD/models:/models:ro q-volta

# CPU-only image (no CUDA libraries)
docker build -t q-volta:cpu --target cpu .
docker run --rm -p 8080:8080 -v $PWD/models:/models:ro q-volta:cpu

# compose
MODELS=./models docker compose up --build                 # GPU
MODELS=./models docker compose --profile cpu up --build   # CPU, port 8081
```

The entrypoint is `qv_server --host 0.0.0.0 --port 8080`; any further flag replaces the default `CMD`, for example
`docker run ... q-volta --model /models/model.qv --tokenizer /models/tokenizer.json --backend cuda --kv int4 --max-ctx 250100 --prefill-chunk 512`
(see [SERVER.md](SERVER.md) and [SERVER_COMPAT.md](SERVER_COMPAT.md); llama-server style `LLAMA_ARG_*` environment variables are accepted).
`curl localhost:8080/health` is the container health check. The kernels are compiled for sm_70 only; other GPU generations need a change in `CMakeLists.txt`.
