#!/usr/bin/env bash
# Idempotent provisioning of the Le Chonk freeciv harness environment
# from a clean host. Safe to re-run: existing resources are detected
# and reused, never silently destroyed.
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE_TAG="freeciv/freeciv-web:latest"
IMAGE_DIGEST="civrealm/freeciv-web@sha256:4923253876a76d3c5f55c6007e530f6c3cd3120b9e2f37a8e943faf920cb373a"
CONTAINER="freeciv-web"
PUBSTATUS="http://localhost:8080/pubstatus"

for tool in docker git uv npm npx curl; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing dependency: $tool" >&2; exit 1; }
done

# Use docker directly when the current user has access; otherwise go
# through a docker-group session.
dock() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  else
    sg docker -c "$(printf '%q ' docker "$@")"
  fi
}

# Report whether a host TCP port has a listener (ss preferred; a loopback
# connect probe as fallback). Existing listeners are never stopped here.
port_busy() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    [ -n "$(ss -ltnH "sport = :$port")" ]
  else
    timeout 1 bash -c ">/dev/tcp/127.0.0.1/$port" 2>/dev/null
  fi
}

# FIRST provisioning step (plan WP0): vendor/civrealm is a git submodule
# of this repository, recorded in .gitmodules and pointing at the
# pavelpenev/civrealm mirror over HTTPS. Initialize it when absent, then
# verify the checked-out HEAD against the recorded gitlink and the
# harness's known-good history: dev base 1b144d6 and the civgov-fixes
# head 36ae456 must both be ancestors. A checkout that already matches
# the recorded gitlink is left untouched, never modified.
echo "==> Vendored civrealm submodule"
CIVREALM_BASE="1b144d6"
CIVREALM_HEAD="36ae456"
if [ -e vendor/civrealm ] && [ ! -d vendor/civrealm/.git ]; then
  echo "vendor/civrealm exists but is not a git checkout; refusing to touch it" >&2
  exit 1
fi
if [ ! -d vendor/civrealm/.git ]; then
  git submodule update --init --recursive vendor/civrealm
fi
CIVREALM_GITLINK="$(git rev-parse ':vendor/civrealm')"
CIVREALM_HEAD_SHA="$(git -C vendor/civrealm rev-parse HEAD)"
if [ "$CIVREALM_HEAD_SHA" != "$CIVREALM_GITLINK" ]; then
  echo "vendor/civrealm HEAD $CIVREALM_HEAD_SHA does not match the recorded gitlink $CIVREALM_GITLINK" >&2
  echo "(run: git submodule update --init --recursive vendor/civrealm)" >&2
  exit 1
fi
git -C vendor/civrealm merge-base --is-ancestor "$CIVREALM_BASE" HEAD \
  || { echo "dev base $CIVREALM_BASE is not an ancestor of the vendored HEAD" >&2; exit 1; }
git -C vendor/civrealm merge-base --is-ancestor "$CIVREALM_HEAD" HEAD \
  || { echo "civgov-fixes head $CIVREALM_HEAD is not an ancestor of the vendored HEAD" >&2; exit 1; }
echo "vendor/civrealm is checked out at $CIVREALM_HEAD_SHA (gitlink $CIVREALM_GITLINK)"

echo "==> Host architecture"
arch="$(uname -m)"
if [ "$arch" != "x86_64" ] && [ "$arch" != "amd64" ]; then
  echo "host architecture is $arch; the pinned image digest is amd64-only" >&2
  exit 1
fi

echo "==> Docker image"
if ! dock image inspect "$IMAGE_DIGEST" >/dev/null 2>&1; then
  dock pull "$IMAGE_DIGEST"
fi
dock tag "$IMAGE_DIGEST" "$IMAGE_TAG"

echo "==> Container"
if dock ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  if [ "$(dock inspect -f '{{.State.Running}}' "$CONTAINER")" = "true" ]; then
    echo "container '$CONTAINER' already running; leaving it as is"
  else
    echo "starting existing stopped container '$CONTAINER'"
    dock start "$CONTAINER"
  fi
else
  echo "==> Port collision check"
  busy=""
  for port in 8080 4002 8888 1240 $(seq 6000 6009) $(seq 7000 7009); do
    if port_busy "$port"; then
      busy="$busy $port"
    fi
  done
  if [ -n "$busy" ]; then
    echo "mapped host ports already in use:$busy" >&2
    echo "existing listeners were not stopped; free the ports and re-run" >&2
    exit 1
  fi
  echo "first start: clean-volume (the container mounts no named volumes)"
  dock run -d --name freeciv-web --user docker -p 8080:80 -p 4002:4002 -p 6000-6009:6000-6009 -p 7000-7009:7000-7009 -p 8888:8080 -p 1240:1240 freeciv/freeciv-web:latest sleep infinity
fi

echo "==> Waiting for stack boot at $PUBSTATUS"
up=0
for _ in $(seq 1 60); do
  if curl -fsS --max-time 2 "$PUBSTATUS" >/dev/null 2>&1; then
    up=1
    break
  fi
  sleep 2
done
if [ "$up" -ne 1 ]; then
  echo "stack did not answer $PUBSTATUS within 120 s" >&2
  exit 1
fi
echo "stack is up"

echo "==> In-container save-path writability"
SAVEGAMES_DIR="/var/lib/tomcat10/webapps/data/savegames/"
dock exec "$CONTAINER" mkdir -p "$SAVEGAMES_DIR" 2>/dev/null || true
if ! dock exec "$CONTAINER" test -w "$SAVEGAMES_DIR"; then
  echo "in-container save path $SAVEGAMES_DIR is not writable" >&2
  exit 1
fi
echo "save path is writable"

echo "==> Python venv (uv, 3.12)"
# Ray is intentionally NOT installed: absent-ray is the config validated
# by upstream civrealm PR #7.
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python -e vendor/civrealm pytest

echo "==> Node (playwright@1.49.0 pinned by package.json)"
npm install
npx playwright install chromium

echo "==> Provisioning complete"
