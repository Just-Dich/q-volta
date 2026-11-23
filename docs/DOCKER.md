# Docker

Weights are not baked into the image: mount a directory with one `.qv` file and `tokenizer.json` at `/models`.

## What was and was not verified
The machine this was written on has **no container runtime**, so `docker build` itself has not been run by the authors. What was checked:
- The two build recipes in the Dockerfile (CUDA and CPU-only) were run natively on a clean `git archive` copy of the repository (no `.git`, no build dirs),
  with the same cmake commands and targets (`qv_server qv_run qv_tok` / `qv_server qv_tok qv_run_cpu`): both build.
- The CUDA-built `qv_server` was started on the V100 with the real `qwen38-27b-e1.qv` and `--kv int4`; `/health` returned `ok` and a chat completion worked.
  Its only shared libraries are `libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12` plus libstdc++/libc, all shipped by `nvidia/cuda:12.9.1-runtime-ubuntu24.04`; the CPU binary needs only libstdc++/libc.
- The entrypoint script was run against a stub in 7 cases (no model, first `*.qv`, `model.qv` preferred, explicit `--model`/`-m`, `LLAMA_ARG_MODEL`, `QV_PORT`).
- Base image tags `nvidia/cuda:12.9.1-{devel,runtime}-ubuntu24.04` exist on Docker Hub (checked via the Hub API).
`scripts/docker_smoke.sh` automates build + run + request for whoever has Docker; please report problems.

## Use
```bash
hf download q-project/Qwen3.8-27B-QVolta qwen38-27b-e1.qv tokenizer.json --local-dir models

# GPU image (Tesla V100 / sm_70 kernels; needs the NVIDIA container toolkit)
docker build -t q-volta .
docker run --rm --gpus all -p 8080:8080 -v $PWD/models:/models:ro q-volta

# CPU-only image (no CUDA libraries)
docker build -t q-volta:cpu --target cpu .
docker run --rm -p 8080:8080 -v $PWD/models:/models:ro q-volta:cpu

# compose
MODELS=./models docker compose up --build                 # GPU, port 8080
MODELS=./models docker compose --profile cpu up --build   # CPU, port 8081

scripts/docker_smoke.sh cuda ./models                      # build + health + one completion
```

## Behaviour
- Entrypoint (`docker/entrypoint.sh`): runs `qv_server --host 0.0.0.0 --port ${QV_PORT:-8080}`; if no `--model`/`-m`/`LLAMA_ARG_MODEL` is given it uses `/models/model.qv`
  or the first `/models/*.qv`; if no `--tokenizer` is given and `/models/tokenizer.json` exists it uses that. Everything after the image name is passed to `qv_server`
  and replaces the default `CMD`, e.g. `docker run ... q-volta --backend cuda --kv int4 --max-ctx 250100 --prefill-chunk 512 --mtp --mtp-depth 2`.
  Flags: [SERVER.md](SERVER.md), [SERVER_COMPAT.md](SERVER_COMPAT.md); llama-server style `LLAMA_ARG_*` variables work.
- Defaults: GPU image `--backend cuda --kv int4 --max-ctx 131072` (fits a 16 GB V100 with the 10.4 GiB e1 file); CPU image `--backend cpu --threads 32`.
- Health check: `curl /health` (503 while the model loads; start period 300 s).
- Kernels are compiled for sm_70 only; other GPU generations need a change in `CMakeLists.txt`.
- CPU image: give the container all cores and do not over-subscribe (the engine pins 32 threads); weights are copied into huge pages when available, so memory limits should exceed the model size.
