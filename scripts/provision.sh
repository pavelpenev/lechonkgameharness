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
# A submodule checkout is initialized when vendor/civrealm/.git exists
# as either a directory (full clone) or a file (standard gitfile layout
# pointing at the parent's module gitdir). A vendor/civrealm present
# without any .git is stray content, not a checkout.
if [ -e vendor/civrealm ] && [ ! -e vendor/civrealm/.git ]; then
  # A non-recursive clone creates the empty mountpoint for an uninitialized
  # gitlink.  Let git submodule initialize that directory; reject only real
  # content that is not owned by the gitlink.
  gitlink_mode="$(git ls-files -s -- vendor/civrealm | awk '{print $1}')"
  stray="$(find vendor/civrealm -mindepth 1 -print -quit 2>/dev/null || true)"
  if [ "$gitlink_mode" = "160000" ] && [ -z "$stray" ]; then
    : # safe empty gitlink mountpoint; initialize below
  else
    echo "vendor/civrealm exists but is not an empty gitlink checkout; refusing to touch it" >&2
    exit 1
  fi
fi
if [ ! -e vendor/civrealm/.git ]; then
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
  expected_image_id="$(dock image inspect -f '{{.Id}}' "$IMAGE_DIGEST")"
  actual_image_id="$(dock inspect -f '{{.Image}}' "$CONTAINER")"
  if [ "$actual_image_id" != "$expected_image_id" ]; then
    echo "container '$CONTAINER' uses image $actual_image_id, expected $expected_image_id" >&2
    echo "refusing to reuse a container that does not match the pinned image" >&2
    exit 1
  fi
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

echo "==> Optional local Freeciv reference"
# Missing reference data must not prevent the game from starting.
if [ "${LECHONK_EXTRACT_FREECIV_REFERENCE:-1}" = 1 ]; then
  FREECIV_CONTAINER="$CONTAINER" FREECIV_IMAGE_REF="$IMAGE_DIGEST" \
    scripts/extract_freeciv_reference.sh || {
      if [ "${FREECIV_REFERENCE_REQUIRED:-0}" = 1 ]; then
        exit 1
      fi
      echo "warning: continuing without local Freeciv reference" >&2
    }
else
  echo "local Freeciv reference extraction disabled"
fi

echo "==> Optional local Freeciv manual"
# The generated MediaWiki manual is supplemental to the live ruleset and is
# deliberately optional.  Keep the required switch independent from the
# encyclopedia extractor: operators can require this source even when the
# generic helpdata file is unavailable in the image.
if [ "${LECHONK_EXTRACT_FREECIV_MANUAL:-1}" = 1 ]; then
  FREECIV_CONTAINER="$CONTAINER" FREECIV_IMAGE_REF="$IMAGE_DIGEST" \
    scripts/extract_freeciv_manual.sh || {
      if [ "${FREECIV_MANUAL_REQUIRED:-0}" = 1 ]; then
        exit 1
      fi
      echo "warning: continuing without local Freeciv manual" >&2
    }
elif [ "${FREECIV_MANUAL_REQUIRED:-0}" = 1 ]; then
  echo "required local Freeciv manual extraction is disabled (set LECHONK_EXTRACT_FREECIV_MANUAL=1)" >&2
  exit 1
else
  echo "local Freeciv manual extraction disabled"
fi

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
uv pip install --python .venv/bin/python -e vendor/civrealm -e . pytest
.venv/bin/python -c 'import lechonk; import lechonk.core.runner'

echo "==> Node (playwright@1.49.0 pinned by package.json)"
npm install
npx playwright install chromium

echo "==> Provisioning complete"
