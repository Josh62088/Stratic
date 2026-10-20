#!/bin/sh
# Wait for the GPU, then evict ollama.
#
# WHY THE WAIT
#   A service can start before the NVIDIA driver has finished binding the cards. Without this, the engine
#   comes up with no visible device and fails, then systemd retries in a loop.
#
# WHY THE EVICTION
#   ollama runs in a container on this machine, not as a systemd service, so `systemctl is-active ollama`
#   says inactive while a model holds tens of gigabytes of VRAM. After a reboot it held 25.7 GB across all
#   three cards. Strata needs GPU 2 to itself, so any loaded ollama model is unloaded first. keep_alive 0 is
#   the API's own "unload now"; the model stays on disk and reloads on demand.
#
# Every step is best-effort: if ollama is not running, this is a no-op and must not block startup.
set -e

i=0
while [ $i -lt 60 ]; do
    if nvidia-smi --query-gpu=index --format=csv,noheader >/dev/null 2>&1; then
        break
    fi
    i=$((i + 1))
    sleep 2
done

models=$(curl -s -m 5 http://127.0.0.1:11434/api/ps 2>/dev/null \
    | grep -o '"name":"[^"]*"' | cut -d'"' -f4) || true

for m in $models; do
    curl -s -m 20 -X POST http://127.0.0.1:11434/api/generate \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$m\",\"keep_alive\":0,\"prompt\":\"\"}" >/dev/null 2>&1 || true
done

# give the driver a moment to hand the VRAM back
sleep 3
exit 0
