#!/bin/sh
# Rebuild the vendored llama.cpp web UI (src/server/webui/gzip) from a llama.cpp checkout.
# usage: scripts/vendor_webui.sh /path/to/llama.cpp [workdir] [--no-theme]     (needs node/npm and network for `npm ci`)
# The Q-Volta theme (src/server/webui-theme: q-volta-theme.patch + files/) is applied to the upstream sources before the build;
# --no-theme builds the pristine upstream UI instead. After an upstream update, re-theme by refreshing the patch (see webui-theme/README.md).
set -eu
L=${1:?llama.cpp checkout}
W=${2:-/tmp/qv_webui_build}
THEME=1; [ "${3:-}" = "--no-theme" ] && THEME=0
HERE=$(cd "$(dirname "$0")/.." && pwd)
rm -rf "$W" && mkdir -p "$W"
cp -r "$L/ui" "$W/ui"
if [ "$THEME" = 1 ]; then
  (cd "$W/ui" && patch -p1 --no-backup-if-mismatch < "$HERE/src/server/webui-theme/q-volta-theme.patch" && cp -r "$HERE/src/server/webui-theme/files/." .)
fi
cd "$W/ui/apps/chat"
npm ci
npm run build                      # SvelteKit static adapter -> dist/ (same command llama.cpp's CMake runs)
cp "$HERE/src/server/webui/LICENSE" dist/LICENSE-llama.cpp-webui.txt   # the MIT licence text, linked from the Settings credit
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
        with open(o, "wb") as fo, gzip.GzipFile(filename="", mode="wb", fileobj=fo, compresslevel=6, mtime=0) as g:  # mtime 0: reproducible ETags
            g.write(d)
        with open(o, "r+b") as fo: fo.seek(9); fo.write(b"\x03")  # OS = Unix, like the gzip streams llama.cpp's CMake writes (identical bytes -> identical ETags)
        man.append((rel, hashlib.sha256(d).hexdigest(), len(d)))
man.sort()
open(os.path.join(out, "MANIFEST.sha256"), "w").write("".join(f"{h}  {r}  ({n} bytes)\n" for r, h, n in man))
print(len(man), "files")
PY
