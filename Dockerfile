# Q-Volta inference server image (OpenAI / llama-server compatible API).
#   docker build -t q-volta .                      # CUDA + CPU backends (Tesla V100 / sm_70 kernels)   [default target: cuda]
#   docker build -t q-volta:cpu --target cpu .     # CPU-only image, no CUDA libraries
# Weights are NOT baked in: mount a directory with the .qv file and tokenizer.json at /models (see docs/DOCKER.md).
# Authoring note: the machine this was written on has no container runtime. Every build step below was run natively on a clean
# `git archive` copy of the repository (CPU and CUDA configurations, same cmake targets), and the entrypoint was run against the
# mock backend; the image itself still has to be built by someone with Docker (docs/DOCKER.md).

ARG CUDA_VERSION=12.9.1
ARG UBUNTU=ubuntu24.04

# ---------------------------------------------------------------- build (CUDA)
FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS build-cuda
RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build g++ python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
# kernels are compiled for sm_70 (set in CMakeLists.txt); no GPU is needed at build time
RUN cmake -S . -B /out -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DQV_CPU=OFF \
 && cmake --build /out --target qv_server qv_run qv_tok

# ---------------------------------------------------------------- build (CPU only)
FROM ubuntu:24.04 AS build-cpu
RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build g++ python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -S . -B /out -G Ninja -DCMAKE_BUILD_TYPE=Release -DQV_CUDA=OFF \
 && cmake --build /out --target qv_server qv_tok qv_run_cpu

# ---------------------------------------------------------------- runtime: CPU only
FROM ubuntu:24.04 AS cpu
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build-cpu /out/qv_server /out/qv_tok /usr/local/bin/
COPY --from=build-cpu /out/src/backends/cpu/qv_run_cpu /usr/local/bin/qv_run
COPY docker/entrypoint.sh /usr/local/bin/qv-entrypoint
RUN chmod +x /usr/local/bin/qv-entrypoint
VOLUME /models
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 CMD curl -fsS http://127.0.0.1:8080/health || exit 1
ENTRYPOINT ["qv-entrypoint"]
CMD ["--backend", "cpu", "--threads", "32"]

# ---------------------------------------------------------------- runtime: CUDA (default target)
FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU} AS cuda
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build-cuda /out/qv_server /out/qv_run /out/qv_tok /usr/local/bin/
COPY docker/entrypoint.sh /usr/local/bin/qv-entrypoint
RUN chmod +x /usr/local/bin/qv-entrypoint
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
VOLUME /models
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 CMD curl -fsS http://127.0.0.1:8080/health || exit 1
ENTRYPOINT ["qv-entrypoint"]
# 10.4 GiB e1 weights + int4 KV: 131072 tokens fit a 16 GB V100 (up to ~250K with --max-ctx 250100 --prefill-chunk 512)
CMD ["--backend", "cuda", "--kv", "int4", "--max-ctx", "131072"]
