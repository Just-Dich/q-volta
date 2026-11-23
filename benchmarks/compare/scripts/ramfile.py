#!/usr/bin/env python3
"""Hold an anonymous RAM-backed file (memfd) so a large intermediate (Q8_0 GGUF) does not use scarce disk.
Usage: ramfile.py NAME   -> prints /proc/<pid>/fd/<fd>, writes it to /tmp/ramfile_NAME.path, then sleeps forever.
Other processes open/write/mmap that path like a normal file (memfd content lives until this process exits)."""
import os, sys, time
name = sys.argv[1]
fd = os.memfd_create(name)
p = f"/proc/{os.getpid()}/fd/{fd}"
open(f"/tmp/ramfile_{name}.path", "w").write(p)
print(p, flush=True)
while True:
    time.sleep(3600)
