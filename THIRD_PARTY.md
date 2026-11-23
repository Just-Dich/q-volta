# Third-party material

| Component | Where | Licence | Upstream / version |
|---|---|---|---|
| llama.cpp web UI ("Llama Chat", SvelteKit app + bundled npm dependencies), vendored as pre-gzipped static build output; also the reference for the llama-server HTTP behaviour that `qv_server` reproduces (no llama.cpp source code is copied into the server) | `src/server/webui/` | MIT, Copyright (c) 2023-2026 The ggml authors | https://github.com/ggml-org/llama.cpp commit 9d6f04f, directory `ui/`; licence text in `src/server/webui/LICENSE`; provenance and rebuild steps in `src/server/webui/README.md` |
