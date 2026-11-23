#!/bin/sh
# Rebuild the vendored llama.cpp web UI (src/server/webui/gzip) from a llama.cpp checkout.
# usage: scripts/vendor_webui.sh /path/to/llama.cpp [workdir]     (needs node/npm and network for `npm ci`)
set -eu
L=${1:?llama.cpp checkout}
W=${2:-/tmp/qv_webui_build}
HERE=$(cd "$(dirname "$0")/.." && pwd)
rm -rf "$W" && mkdir -p "$W"
cp -r "$L/ui" "$W/ui"
cd "$W/ui/apps/chat"
npm ci
npm run build                      # SvelteKit static adapter -> dist/ (same command llama.cpp's CMake runs)
OUT="$HERE/src/server/webui"
python3 -I - "$W/ui/apps/chat/dist" "$OUT" <<'PY'
import os, sys, gzip, hashlib, shutil
src, out = sys.argv[1:3]
dst = os.path.join(out, "gzip")
shutil.rmtree(dst, ignore_errors=True)
man = []
for root, _, fs in os.walk(src):
    for f in fs:
        p = os.path.join(root, f); rel = os.path.relpath(p, src)
        d = open(p, "rb").read()
        o = os.path.join(dst, rel); os.makedirs(os.path.dirname(o), exist_ok=True)
        with open(o, "wb") as fo, gzip.GzipFile(filename="", mode="wb", fileobj=fo, compresslevel=9, mtime=0) as g:  # mtime 0: reproducible ETags
            g.write(d)
        man.append((rel, hashlib.sha256(d).hexdigest(), len(d)))
man.sort()
open(os.path.join(out, "MANIFEST.sha256"), "w").write("".join(f"{h}  {r}  ({n} bytes)\n" for r, h, n in man))
print(len(man), "files")
PY
