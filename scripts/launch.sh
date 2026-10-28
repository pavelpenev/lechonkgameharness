#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

# Keep --help useful on a brand-new checkout without requiring a venv.
for arg in "$@"; do
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
    cat <<'EOF'
Usage: ./scripts/launch.sh [--turns N | --until-game-over] [--run-id ID] [--seed N] [options]

Provision the local Freeciv stack when needed, then run Mistral through the
model harness. A timestamped run ID is generated unless --run-id is supplied.
Progress and the actual observer URL are printed once the game session starts.
Pass --open-observer to ask the system browser to open that read-only observer
URL once; without it, no browser is started. The webclient may still require
clicking "Observe Game".
Interactive defaults are --api-timeout-s 900, --acting-deadline-s 1200, and
--stream. These are upper bounds, not expected wait times. Pass --no-stream to
use complete responses instead. With no horizon flag, the run continues until
Freeciv reports game over; pass --turns for a bounded smoke test. Interactive
runs use --context-policy conversation_v1, which retains the ordered model
conversation and compacts it at 500,000 estimated tokens; pass
--context-policy legacy to disable conversation history.
Interactive runs use the gameplay_v2 prompt profile, which can make bounded
read-only lookups against the active ruleset snapshot. The opt-in gameplay_v3
profile adds the versioned candidate surface; gameplay_v4 is a separate frozen
strategic prompt bundle. gameplay_v5 is an experimental v4-derived bundle
with one bounded decision-pass instruction; it is not proven faster or better.
Pass --prompt-profile gameplay_v3, gameplay_v4, or gameplay_v5 for fresh runs,
or legacy_eval to compare against evaluation prompts. All profiles leave model
output uncapped by the harness.
EOF
    exit 0
  fi
done

# Use Docker directly when the current user has access; otherwise go through
# a docker-group session.  Keep the probe inside the helper so every health
# check uses the same access path (and so a shell without docker-group access
# does not make a healthy stack look unavailable).
dock() {
  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    docker "$@"
  else
    sg docker -c "$(printf '%q ' docker "$@")"
  fi
}

stack_ready=0
if [[ -x .venv/bin/python && -d vendor/civrealm && -f vendor/civrealm/pyproject.toml ]]; then
  image_ok=0
  if dock info >/dev/null 2>&1 \
      && [ "$(dock inspect -f '{{.State.Running}}' freeciv-web 2>/dev/null)" = true ] \
      && expected_image_id="$(dock image inspect -f '{{.Id}}' 'civrealm/freeciv-web@sha256:4923253876a76d3c5f55c6007e530f6c3cd3120b9e2f37a8e943faf920cb373a' 2>/dev/null)" \
      && actual_image_id="$(dock inspect -f '{{.Image}}' freeciv-web 2>/dev/null)" \
      && [ "$expected_image_id" = "$actual_image_id" ]; then
    image_ok=1
  fi
  vendor_ok=0
  gitlink_sha="$(git rev-parse ':vendor/civrealm' 2>/dev/null || true)"
  vendor_head_sha="$(git -C vendor/civrealm rev-parse HEAD 2>/dev/null || true)"
  if [ -n "$gitlink_sha" ] && [ "$vendor_head_sha" = "$gitlink_sha" ]; then
    vendor_ok=1
  fi
  if ./.venv/bin/python -c 'import lechonk, mistralai' >/dev/null 2>&1 \
      && command -v curl >/dev/null 2>&1 \
      && curl -fsS --max-time 2 http://localhost:8080/pubstatus >/dev/null 2>&1 \
      && [ "$image_ok" -eq 1 ] && [ "$vendor_ok" -eq 1 ]; then
    stack_ready=1
  fi
fi

if (( stack_ready == 1 )); then
  echo "==> Existing Freeciv stack and Python environment are healthy; reusing them"
else
  echo "==> Checking/provisioning the Freeciv stack and Python environment"
  ./scripts/provision.sh
fi

# Provisioning is intentionally skipped for a healthy existing stack, but the
# local encyclopedia may still be absent after an upgrade or a prior manual
# start. Populate the fixed path used by the session adapter without making a
# normal game launch depend on Docker reference extraction.
reference_dir="$repo_dir/artifacts/reference/freeciv"
reference_path="$reference_dir/reference.json"
reference_ruleset="$reference_dir/classic"
# Keep the canonical path visible for operators and compatibility checks.
helpdata_path="$repo_dir/artifacts/reference/freeciv/helpdata.txt"
if [ "${LECHONK_EXTRACT_FREECIV_REFERENCE:-1}" = 1 ] && [ -x "$repo_dir/scripts/extract_freeciv_reference.sh" ]; then
  if [ -f "$reference_path" ] && [ -d "$reference_ruleset" ]; then
    echo "==> Verifying local Freeciv reference: $reference_path"
  else
    echo "==> Local Freeciv reference missing; attempting extraction"
  fi
  if ! FREECIV_CONTAINER=freeciv-web \
      FREECIV_IMAGE_REF='civrealm/freeciv-web@sha256:4923253876a76d3c5f55c6007e530f6c3cd3120b9e2f37a8e943faf920cb373a' \
      "$repo_dir/scripts/extract_freeciv_reference.sh"; then
    if [ "${FREECIV_REFERENCE_REQUIRED:-0}" = 1 ]; then
      echo "required local Freeciv reference could not be extracted" >&2
      exit 1
    fi
    echo "warning: continuing without local Freeciv reference" >&2
  fi
else
  echo "==> Local Freeciv reference extraction disabled or unavailable"
fi

# Provisioning is skipped on a healthy-stack reuse path, so make the
# supplemental generated manual converge here as well.  The extractor is
# checksum/idempotence guarded: a valid artifact is verified and returned
# without invoking the expensive in-container generator.
manual_dir="$repo_dir/artifacts/reference/freeciv-manual"
manual_manifest="$manual_dir/reference.json"
if [ "${LECHONK_EXTRACT_FREECIV_MANUAL:-1}" = 1 ] && [ -x "$repo_dir/scripts/extract_freeciv_manual.sh" ]; then
  if [ -f "$manual_manifest" ]; then
    echo "==> Verifying local Freeciv manual: $manual_manifest"
  else
    echo "==> Local Freeciv manual missing; attempting generation"
  fi
  if ! FREECIV_CONTAINER=freeciv-web \
      FREECIV_IMAGE_REF='civrealm/freeciv-web@sha256:4923253876a76d3c5f55c6007e530f6c3cd3120b9e2f37a8e943faf920cb373a' \
      "$repo_dir/scripts/extract_freeciv_manual.sh"; then
    if [ "${FREECIV_MANUAL_REQUIRED:-0}" = 1 ]; then
      echo "required local Freeciv manual could not be generated" >&2
      exit 1
    fi
    echo "warning: continuing without local Freeciv manual" >&2
  fi
elif [ "${FREECIV_MANUAL_REQUIRED:-0}" = 1 ]; then
  echo "required local Freeciv manual extraction is disabled or unavailable" >&2
  exit 1
else
  echo "==> Local Freeciv manual extraction disabled or unavailable"
fi

run_id_present=0
for arg in "$@"; do
  case "$arg" in
    --run-id|--run-id=*) run_id_present=1 ;;
  esac
done

run_args=("$@")
if (( run_id_present == 0 )); then
  run_args+=(--run-id "interactive-$(date -u +%Y%m%dT%H%M%S%NZ)-$$")
fi

horizon_present=0
for arg in "${run_args[@]}"; do
  case "$arg" in
    --turns|--turns=*|--until-game-over) horizon_present=1 ;;
  esac
done
if (( horizon_present == 0 )); then
  run_args+=(--until-game-over)
fi

api_timeout_present=0
acting_deadline_present=0
for arg in "${run_args[@]}"; do
  case "$arg" in
    --api-timeout-s|--api-timeout-s=*) api_timeout_present=1 ;;
    --acting-deadline-s|--acting-deadline-s=*) acting_deadline_present=1 ;;
  esac
done
if (( api_timeout_present == 0 )); then
  run_args+=(--api-timeout-s 900)
fi
if (( acting_deadline_present == 0 )); then
  run_args+=(--acting-deadline-s 1200)
fi

stream_present=0
for arg in "${run_args[@]}"; do
  case "$arg" in
    --stream|--no-stream) stream_present=1 ;;
  esac
done
if (( stream_present == 0 )); then
  run_args+=(--stream)
fi

context_policy_present=0
for arg in "${run_args[@]}"; do
  case "$arg" in
    --context-policy|--context-policy=*) context_policy_present=1 ;;
  esac
done
if (( context_policy_present == 0 )); then
  run_args+=(--context-policy conversation_v1)
fi

prompt_profile_present=0
for arg in "${run_args[@]}"; do
  case "$arg" in
    --prompt-profile|--prompt-profile=*) prompt_profile_present=1 ;;
  esac
done
if (( prompt_profile_present == 0 )); then
  run_args+=(--prompt-profile gameplay_v2)
fi

echo
echo "The terminal will show turn, model-call, decision, and action progress."
echo "The actual Freeciv observer URL will be printed when the game session starts."
echo

exec ./.venv/bin/python scripts/wp2_model_run.py "${run_args[@]}"
