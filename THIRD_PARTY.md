# Third-party code and data

| component | where | licence | use |
|---|---|---|---|
| `stb_image.h` v2.34 (Sean Barrett, https://github.com/nothings/stb) | `third_party/stb/stb_image.h` (unmodified, downloaded from the upstream master branch) | public domain / MIT (dual: "ALTERNATIVE A - MIT License", "ALTERNATIVE B - Public Domain (www.unlicense.org)", text at the end of the header) | PNG / JPEG / BMP / GIF / TGA / PSD / PNM / PIC decoding for image input (`src/vision/image.cpp`, compiled with `STBI_NO_STDIO`, `STBI_NO_LINEAR`, `STBI_NO_HDR`, `STBI_MAX_DIMENSIONS 16384`) |
| llama.cpp web UI ("Llama Chat", SvelteKit app + bundled npm dependencies), vendored as pre-gzipped static build output, rebuilt with a Q-Volta theme patch (`src/server/webui-theme/`: colours, logo, strings; the upstream MIT licence and the credit "Based on the llama.cpp web UI (MIT)" are kept in the UI Settings and as `/LICENSE-llama.cpp-webui.txt`; the Llama fonts are no longer bundled); also the reference for the llama-server HTTP behaviour that `qv_server` reproduces (no llama.cpp source code is copied into the server) | `src/server/webui/` | MIT, Copyright (c) 2023-2026 The ggml authors | https://github.com/ggml-org/llama.cpp commit 9d6f04f, directory `ui/`; licence text in `src/server/webui/LICENSE`; provenance and rebuild steps in `src/server/webui/README.md` |

Everything else in this repository is written from scratch (see README "Third-party"). The Hugging Face `transformers` PR #51050 sources (Apache-2.0) were read as the reference
for the Inkling vision tower and run as an external process for golden data (`tools/hfparity/`); no code was copied. Model weights carry their own licences.
