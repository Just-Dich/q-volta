# Vendored llama.cpp web UI ("Llama Chat")

* Source: https://github.com/ggml-org/llama.cpp, directory `ui/` (SvelteKit app `ui/apps/chat` + `ui/lib/ui`), commit **9d6f04f**
  ("ui: show vision as glasses and audio as headphones (#32617)"), the same commit qv_server's API is matched against.
* Licence: MIT, Copyright (c) 2023-2026 The ggml authors (`LICENSE`, copied from the llama.cpp repository root; the app's own `ui/apps/chat/LICENSE` is "Copyright (c) ggml").
  The bundle also contains its npm dependencies (Svelte, bits-ui, marked, highlight.js, katex, dexie, ... see `package-lock.llama-chat.json` for the exact versions); all are permissively licensed (MIT/ISC/Apache-2.0/BSD) as reported by the upstream lock file.
* Build: built from that exact source with `npm ci && npm run build` (node v24, exactly what llama.cpp's `scripts/ui-assets.cmake` runs when `LLAMA_BUILD_UI=ON`), NOT downloaded
  from a release. `scripts/vendor_webui.sh /path/to/llama.cpp` reproduces it. The build stamps `build.json` with `{"version":"b0000"}` (no llama.cpp build number outside a CI build).
* `gzip/` holds every file of `dist/` (96 files) compressed with `gzip -9` and a zero timestamp, under its served path. They are embedded into `qv_server` by
  `cmake/qv_embed_webui.cmake` (generated `webui_assets.cpp` in the build directory), served with `Content-Encoding: gzip`, so the server has no runtime file dependency.
  The ETag of a file is the quoted SHA-256 of its gzip bytes (as in llama-server). `MANIFEST.sha256` lists the SHA-256 of the uncompressed files.
