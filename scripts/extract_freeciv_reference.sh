#!/usr/bin/env bash
# Extract Freeciv's bundled encyclopedia and classic ruleset safely.
set -euo pipefail
cd "$(dirname "$0")/.."

CONTAINER="${FREECIV_CONTAINER:-freeciv-web}"
IMAGE_REF="${FREECIV_IMAGE_REF:-civrealm/freeciv-web@sha256:4923253876a76d3c5f55c6007e530f6c3cd3120b9e2f37a8e943faf920cb373a}"
OUTPUT_DIR="${FREECIV_REFERENCE_DIR:-artifacts/reference/freeciv}"
REQUIRED="${FREECIV_REFERENCE_REQUIRED:-0}"
REQUIRE_HELP="${FREECIV_REFERENCE_REQUIRE_HELP:-0}"
DIAGNOSE=0
if [ "${1:-}" = "--diagnose" ]; then
  DIAGNOSE=1
  shift
fi
if [ "$#" -ne 0 ]; then
  echo "usage: $0 [--diagnose]" >&2
  exit 2
fi

fail_or_warn() {
  if [ "$REQUIRED" = 1 ]; then
    echo "freeciv reference extraction failed: $1" >&2
    exit 1
  fi
  echo "warning: freeciv reference unavailable: $1" >&2
  exit 0
}

docker_cmd() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  elif command -v sg >/dev/null 2>&1; then
    sg docker -c "$(printf '%q ' docker "$@")"
  else
    return 125
  fi
}

tree_sha256() {
  (cd "$1" && find . -type f -print0 | sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}')
}

has_symlinks() {
  [ -n "$(find "$1" -type l -print -quit 2>/dev/null)" ]
}

command -v docker >/dev/null 2>&1 || fail_or_warn "Docker CLI is not installed"
docker_cmd info >/dev/null 2>&1 || fail_or_warn "Docker API is unavailable (check the daemon/socket permissions)"
docker_cmd inspect "$CONTAINER" >/dev/null 2>&1 || fail_or_warn "container '$CONTAINER' is not available through the Docker API"
if [ "$DIAGNOSE" = 1 ]; then
  echo "Freeciv reference diagnostic: container=$CONTAINER"
  echo "image=$(docker_cmd inspect -f '{{.Image}}' "$CONTAINER" 2>/dev/null || echo unavailable)"
  docker_cmd exec "$CONTAINER" sh -c '
    found=0
    for scan_root in /home/docker/freeciv /usr/share/freeciv /usr/local/share/freeciv /opt/freeciv /usr/local/freeciv; do
      [ -d "$scan_root" ] || continue
      files="$(find "$scan_root" -maxdepth 8 -type f \( -name helpdata.txt -o -name freeciv-version -o -name VERSION \) -print 2>/dev/null | head -n 100)"
      if [ -n "$files" ]; then
        printf '%s\n' "$files" | while IFS= read -r path; do printf "file=%s\n" "$path"; done
        found=1
      fi
      classic_dirs="$(find "$scan_root" -maxdepth 7 -type d -name classic -print 2>/dev/null | head -n 100)"
      if [ -n "$classic_dirs" ]; then
        printf '%s\n' "$classic_dirs" | while IFS= read -r path; do printf "classic_dir=%s\n" "$path"; done
        found=1
      fi
    done
    [ "$found" = 1 ] || echo "no matching files or classic directories under supported prefixes"
    for p in /home/docker/freeciv/share/freeciv/version.txt /usr/share/freeciv/version.txt /usr/local/share/freeciv/version.txt; do
      if [ -f "$p" ]; then printf "version_file=%s\n" "$p"; sed -n "1p" "$p"; fi
    done
  ' || fail_or_warn "Docker exec failed while diagnosing container '$CONTAINER'"
  exit 0
fi
expected_image_id="$(docker_cmd image inspect -f '{{.Id}}' "$IMAGE_REF" 2>/dev/null || true)"
actual_image_id="$(docker_cmd inspect -f '{{.Image}}' "$CONTAINER" 2>/dev/null || true)"
[ -n "$expected_image_id" ] || fail_or_warn "pinned image '$IMAGE_REF' is not available locally"
[ -n "$actual_image_id" ] || fail_or_warn "could not inspect image identity for container '$CONTAINER'"
[ "$actual_image_id" = "$expected_image_id" ] || fail_or_warn "container '$CONTAINER' does not use pinned image '$IMAGE_REF'"
image_version="$(docker_cmd image inspect -f '{{index .Config.Labels "org.opencontainers.image.version"}}' "$IMAGE_REF" 2>/dev/null || true)"
[ -n "$image_version" ] || image_version="unknown (pinned image digest)"

# Prefer known image locations, then search only bounded Freeciv prefixes.
help_path="$(docker_cmd exec "$CONTAINER" sh -c '
  for p in /home/docker/freeciv/share/freeciv/helpdata.txt /usr/share/freeciv/helpdata.txt /usr/local/share/freeciv/helpdata.txt; do
    [ -f "$p" ] && { printf "%s\n" "$p"; exit 0; }
  done
  for root in /home/docker/freeciv /usr/share/freeciv /usr/local/share/freeciv; do
    if [ -d "$root" ]; then
      found="$(find "$root" -maxdepth 6 -type f -name helpdata.txt -print -quit 2>/dev/null)"
      if [ -n "$found" ]; then
        printf "%s\n" "$found"
        exit 0
      fi
    fi
  done
  exit 1
' 2>/dev/null)" || true
if [ -z "$help_path" ]; then
  if [ "$REQUIRE_HELP" = 1 ]; then
    echo "freeciv reference extraction failed: helpdata.txt was not found in supported Freeciv prefixes (required by FREECIV_REFERENCE_REQUIRE_HELP=1)" >&2
    exit 1
  fi
  echo "note: bundled helpdata.txt was not found; continuing with the classic ruleset only. The generated manual, when extracted, is available under artifacts/reference/freeciv-manual." >&2
fi

classic_path="$(docker_cmd exec "$CONTAINER" sh -c '
  p="$1"
  for candidate in "${p%/*}/classic" /home/docker/freeciv/share/freeciv/classic /usr/share/freeciv/classic /usr/local/share/freeciv/classic /opt/freeciv/share/freeciv/classic /usr/local/freeciv/share/freeciv/classic; do
    [ -d "$candidate" ] && { printf "%s\n" "$candidate"; exit 0; }
  done
  exit 1
' sh "$help_path" 2>/dev/null)" || true
[ -n "$classic_path" ] || fail_or_warn "classic ruleset directory was not found in supported Freeciv prefixes"

parent_dir="$(dirname "$OUTPUT_DIR")"
mkdir -p "$parent_dir"
if [ -e "$OUTPUT_DIR" ]; then
  manifest="$OUTPUT_DIR/reference.json"
  if [ -f "$manifest" ] && [ -d "$OUTPUT_DIR/classic" ]; then
    manifest_image="$(sed -n 's/.*"image_ref": "\([^"]*\)".*/\1/p' "$manifest")"
    manifest_help_status="$(sed -n 's/.*"helpdata_status": "\([^"]*\)".*/\1/p' "$manifest")"
    manifest_help_sha="$(sed -n 's/.*"helpdata_sha256": "\([^"]*\)".*/\1/p' "$manifest")"
    manifest_ruleset_sha="$(sed -n 's/.*"classic_ruleset_tree_sha256": "\([^"]*\)".*/\1/p' "$manifest")"
    if has_symlinks "$OUTPUT_DIR/classic"; then
      fail_or_warn "existing classic ruleset contains symlinks; refusing to trust or replace it"
    fi
    current_ruleset_sha="$(tree_sha256 "$OUTPUT_DIR/classic")"
    help_valid=0
    if [ "$manifest_help_status" = "unavailable" ] && [ ! -e "$OUTPUT_DIR/helpdata.txt" ]; then
      help_valid=1
    elif [ "$manifest_help_status" = "available" ] && [ -f "$OUTPUT_DIR/helpdata.txt" ]; then
      current_help_sha="$(sha256sum "$OUTPUT_DIR/helpdata.txt" | awk '{print $1}')"
      [ "$manifest_help_sha" = "$current_help_sha" ] && help_valid=1
    elif [ -z "$manifest_help_status" ] && [ -f "$OUTPUT_DIR/helpdata.txt" ]; then
      # Manifests written before helpdata_status was introduced remain valid.
      current_help_sha="$(sha256sum "$OUTPUT_DIR/helpdata.txt" | awk '{print $1}')"
      [ -n "$manifest_help_sha" ] && [ "$manifest_help_sha" = "$current_help_sha" ] && help_valid=1
    fi
    if [ "$manifest_image" = "$IMAGE_REF" ] && [ "$manifest_ruleset_sha" = "$current_ruleset_sha" ] && [ "$help_valid" = 1 ]; then
      echo "freeciv reference verified at $OUTPUT_DIR; leaving it untouched"
      exit 0
    fi
  fi
  fail_or_warn "existing reference at $OUTPUT_DIR failed image or checksum verification; inspect reference.json and remove/repair it before retrying"
fi

staging="$(mktemp -d "$parent_dir/.freeciv-reference.XXXXXX")"
cleanup() { rm -rf "$staging"; }
trap cleanup EXIT
mkdir -p "$staging/classic"
if [ -n "$help_path" ]; then
  docker_cmd cp "$CONTAINER:$help_path" "$staging/helpdata.txt" ||
    fail_or_warn "could not copy $help_path from container '$CONTAINER'"
fi
docker_cmd cp "$CONTAINER:$classic_path/." "$staging/classic/" ||
  fail_or_warn "could not copy $classic_path from container '$CONTAINER'"
if has_symlinks "$staging/classic"; then
  fail_or_warn "copied classic ruleset contains symlinks; refusing to save it"
fi

ruleset_sha="$(tree_sha256 "$staging/classic")"
safe_image_ref="$(printf '%s' "$IMAGE_REF" | sed 's/\\/\\\\/g; s/"/\\"/g')"
safe_image_version="$(printf '%s' "$image_version" | sed 's/\\/\\\\/g; s/"/\\"/g')"
safe_classic_path="$(printf '%s' "$classic_path" | sed 's/\\/\\\\/g; s/"/\\"/g')"
if [ -n "$help_path" ]; then
  help_sha="$(sha256sum "$staging/helpdata.txt" | awk '{print $1}')"
  safe_help_path="$(printf '%s' "$help_path" | sed 's/\\/\\\\/g; s/"/\\"/g')"
  help_status="available"
  help_path_json="\"$safe_help_path\""
  help_sha_json="\"$help_sha\""
else
  help_sha=""
  help_status="unavailable"
  help_path_json="null"
  help_sha_json="null"
fi
cat > "$staging/reference.json" <<EOF
{
  "schema_version": "freeciv-reference-v1",
  "image_ref": "$safe_image_ref",
  "image_version": "$safe_image_version",
  "container": "$CONTAINER",
  "helpdata_status": "$help_status",
  "helpdata_path": $help_path_json,
  "classic_ruleset_path": "$safe_classic_path",
  "helpdata_sha256": $help_sha_json,
  "classic_ruleset_tree_sha256": "$ruleset_sha",
  "license": "Freeciv data is GPL-2.0-or-later; retain upstream notices before redistribution."
}
EOF
cat > "$staging/LICENSE.txt" <<'EOF'
This directory contains the classic ruleset, and optionally Freeciv
helpdata.txt, copied from the pinned Freeciv-Web image recorded in
reference.json. Freeciv data
is GPL-2.0-or-later; consult upstream Freeciv-Web licensing before sharing.
EOF

mv "$staging" "$OUTPUT_DIR" ||
  fail_or_warn "destination appeared while extraction was in progress; no files were replaced"
trap - EXIT
if [ -n "$help_sha" ]; then
  echo "extracted Freeciv reference to $OUTPUT_DIR (helpdata $help_sha)"
else
  echo "extracted Freeciv reference to $OUTPUT_DIR (classic ruleset only; helpdata unavailable)"
fi
