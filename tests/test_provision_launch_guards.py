"""Regression tests for the provisioning and launch reuse guards.

These tests deliberately avoid Docker: the clone test exercises the real Git
state produced by a fresh non-recursive checkout, while the reuse assertions
make sure the executable scripts retain their fail-closed checks.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
import json
import importlib.util

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "scripts" / "launch.sh"
PROVISION = ROOT / "scripts" / "provision.sh"
REFERENCE = ROOT / "scripts" / "extract_freeciv_reference.sh"
MANUAL = ROOT / "scripts" / "extract_freeciv_manual.sh"

_LIFECYCLE_SPEC = importlib.util.spec_from_file_location(
    "artifact_lifecycle", ROOT / "scripts" / "artifact_lifecycle.py")
_LIFECYCLE = importlib.util.module_from_spec(_LIFECYCLE_SPEC)
assert _LIFECYCLE_SPEC.loader is not None
_LIFECYCLE_SPEC.loader.exec_module(_LIFECYCLE)


def test_launch_help_is_an_executable_smoke_test() -> None:
    result = subprocess.run(
        [str(LAUNCH), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Usage: ./scripts/launch.sh" in result.stdout
    assert "gameplay_v2" in result.stdout
    assert "gameplay_v3" in result.stdout
    assert "gameplay_v4" in result.stdout


def test_direct_runner_exposes_explicit_prompt_profiles() -> None:
    result = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"),
         str(ROOT / "scripts" / "wp2_model_run.py"), "--help"],
        cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--prompt-profile" in result.stdout
    assert "legacy_eval" in result.stdout
    assert "gameplay_v1" in result.stdout
    assert "gameplay_v2" in result.stdout
    assert "gameplay_v3" in result.stdout
    assert "gameplay_v4" in result.stdout
    assert "--aifill" in result.stdout


@pytest.mark.parametrize("value", ["0", "11", "100"])
def test_direct_runner_rejects_aifill_outside_server_capacity(value: str) -> None:
    result = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"),
         str(ROOT / "scripts" / "wp2_model_run.py"), "--turns", "1",
         "--aifill", value],
        cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert "--aifill must be between 1 and 10" in result.stderr


def test_direct_runner_converts_opponents_to_total_aifill() -> None:
    source = (ROOT / "scripts" / "wp2_model_run.py").read_text()
    assert "args.aifill = args.opponents + 1" in source
    assert '"configured_ai_opponents": AIFILL - 1' in source


def test_fresh_run_collision_and_trace_retry_are_fail_closed() -> None:
    source = (ROOT / "scripts" / "wp2_model_run.py").read_text()
    lifecycle = (ROOT / "scripts" / "artifact_lifecycle.py").read_text()
    assert "claim_run_directory" in source
    assert "fresh run rejected: target run directory already contains" in lifecycle
    assert 'TRACE_PATH = fresh_artifact("model_traces.jsonl")' in source
    assert 'refusing to overwrite prior retry artifact' in lifecycle


def test_fresh_collision_preserves_existing_bytes(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    directory.mkdir()
    prior = directory / "transcript.jsonl"
    prior.write_bytes(b"prior")
    with pytest.raises(_LIFECYCLE.ArtifactCollisionError):
        _LIFECYCLE.ensure_fresh_run_directory(directory, attempt=1,
                                               resuming=False)
    assert prior.read_bytes() == b"prior"


def test_retry_trace_rotation_preserves_attempt_one_and_refuses_backup_overwrite(
        tmp_path: Path) -> None:
    trace = tmp_path / "model_traces.jsonl"
    trace.write_bytes(b"attempt one")
    _LIFECYCLE.prepare_attempt_artifact(trace, attempt=2)
    assert not trace.exists()
    assert (tmp_path / "model_traces.jsonl.attempt1").read_bytes() == b"attempt one"
    trace.write_bytes(b"attempt two")
    with pytest.raises(_LIFECYCLE.ArtifactCollisionError):
        _LIFECYCLE.prepare_attempt_artifact(trace, attempt=2)
    assert trace.read_bytes() == b"attempt two"


def test_empty_provisioning_directory_is_allowed(tmp_path: Path) -> None:
    directory = tmp_path / "empty-run"
    directory.mkdir()
    _LIFECYCLE.ensure_fresh_run_directory(directory, attempt=1,
                                           resuming=False)


def test_run_claim_rejects_symlink_and_preserves_retry_metadata(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(_LIFECYCLE.ArtifactCollisionError):
        _LIFECYCLE.claim_run_directory(link, attempt=1, resuming=False)

    run = tmp_path / "run"
    _LIFECYCLE.claim_run_directory(run, attempt=1, resuming=False)
    (run / "run_meta.json").write_bytes(b"meta")
    (run / "recovery.json").write_bytes(b"recovery")
    checkpoints = run / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "save").write_bytes(b"save")
    _LIFECYCLE.prepare_retry_artifacts(run, attempt=2)
    assert (run / "run_meta.json.attempt1").read_bytes() == b"meta"
    assert (run / "recovery.json.attempt1").read_bytes() == b"recovery"
    assert (run / "checkpoints.attempt1" / "save").read_bytes() == b"save"


def test_retry_token_and_reference_preflight_are_fail_closed(tmp_path: Path) -> None:
    run = tmp_path / "run"
    token = _LIFECYCLE.claim_run_directory(run, attempt=1, resuming=False)
    assert _LIFECYCLE.claim_run_directory(run, attempt=2, resuming=False,
                                          token=token) == token
    with pytest.raises(_LIFECYCLE.ArtifactCollisionError):
        _LIFECYCLE.claim_run_directory(run, attempt=2, resuming=False,
                                       token="forged")
    (run / "ruleset_reference.json").write_bytes(b"reference")
    (run / "ruleset_reference_meta.json").mkdir()
    (run / "token_budget_table.json").write_bytes(b"budget")
    with pytest.raises(_LIFECYCLE.ArtifactCollisionError):
        _LIFECYCLE.prepare_retry_artifacts(run, attempt=2)
    # Preflight failure leaves every source untouched.
    assert (run / "ruleset_reference.json").read_bytes() == b"reference"
    assert (run / "token_budget_table.json").read_bytes() == b"budget"
    assert (run / "ruleset_reference_meta.json").is_dir()


def test_retry_preflights_every_source_before_any_rotation(tmp_path: Path) -> None:
    run = tmp_path / "run"
    token = _LIFECYCLE.claim_run_directory(run, attempt=1, resuming=False)
    names = ["conversation.jsonl", "state.json", "api_calls.jsonl",
             "model_traces.jsonl", "transcript.jsonl",
             "token_budget_table.json", "metrics_report.json",
             "manifest_runtime.json", "run_meta.json", "recovery.json",
             "ruleset_reference.json", "ruleset_reference_meta.json"]
    for name in names:
        (run / name).write_bytes(name.encode())
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "save").write_bytes(b"save")
    # A backup collision anywhere in the set must leave every source intact.
    (run / "model_traces.jsonl.attempt1").write_bytes(b"old backup")
    with pytest.raises(_LIFECYCLE.ArtifactCollisionError):
        _LIFECYCLE.prepare_retry_artifacts(run, attempt=2)
    for name in names:
        assert (run / name).read_bytes() == name.encode()
    assert (run / "checkpoints" / "save").read_bytes() == b"save"
    assert (run / "model_traces.jsonl.attempt1").read_bytes() == b"old backup"
    assert _LIFECYCLE.claim_run_directory(run, attempt=2, resuming=False,
                                          token=token) == token


def test_model_trace_path_is_global_before_callback_writes() -> None:
    source = (ROOT / "scripts" / "wp2_model_run.py").read_text()
    assert "global CONTEXT_JOURNAL_PATH, CONVERSATION_STATE_PATH, RESUME_CHECKPOINT, TRACE_PATH" in source
    assert 'TRACE_PATH = fresh_artifact("model_traces.jsonl")' in source


def test_gameplay_profile_requires_conversation_context() -> None:
    result = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"),
         str(ROOT / "scripts" / "wp2_model_run.py"), "--turns", "1",
         "--prompt-profile", "gameplay_v1"],
        cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert "requires --context-policy conversation_v1" in result.stderr


def test_gameplay_v2_profile_requires_conversation_context() -> None:
    result = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"),
         str(ROOT / "scripts" / "wp2_model_run.py"), "--turns", "1",
         "--prompt-profile", "gameplay_v2"],
        cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert "requires --context-policy conversation_v1" in result.stderr


def test_runner_wires_prompt_profile_into_runtime_config() -> None:
    source = (ROOT / "scripts" / "wp2_model_run.py").read_text()
    config_block = source.split("config = RunnerConfig(", 1)[1].split(
        "runner = GameRunner(", 1)[0]
    assert "prompt_profile=PROMPT_PROFILE" in config_block
    assert "aifill=AIFILL" in config_block


def test_direct_legacy_profile_warns_about_lookup_tool() -> None:
    source = (ROOT / "scripts" / "wp2_model_run.py").read_text()
    assert "has no \"" in source
    assert "encyclopedia lookup tool" in source
    assert "--prompt-profile gameplay_v2" in source


def test_nonrecursive_clone_has_an_empty_gitlink_mountpoint(tmp_path: Path) -> None:
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "--no-recurse-submodules", "--no-hardlinks", str(ROOT), str(clone)],
        check=True,
        text=True,
        capture_output=True,
    )

    mountpoint = clone / "vendor" / "civrealm"
    assert mountpoint.is_dir()
    assert not (mountpoint / ".git").exists()
    mode = subprocess.run(
        ["git", "-C", str(clone), "ls-files", "-s", "--", "vendor/civrealm"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.split()[0]
    assert mode == "160000"
    assert not any(mountpoint.iterdir())


def test_reuse_guards_fail_closed_on_provenance_mismatch() -> None:
    launch = LAUNCH.read_text()
    provision = PROVISION.read_text()

    # Fast-path reuse requires the pinned image and exact fork gitlink checkout.
    assert "dock image inspect -f '{{.Id}}'" in launch
    assert 'printf \'%q \' docker "$@"' in launch
    assert "git rev-parse ':vendor/civrealm'" in launch
    assert "git -C vendor/civrealm apply --reverse --check" not in launch
    assert '"$expected_image_id" = "$actual_image_id"' in launch
    assert '"$vendor_head_sha" = "$gitlink_sha"' in launch
    assert '"$patch_ok" -eq 1' not in launch

    # Provisioning refuses an existing container with a different image; it
    # must not silently remove and recreate it.
    assert "refusing to reuse a container that does not match the pinned image" in provision
    assert 'dock rm "$CONTAINER"' not in provision


def test_reference_extractor_is_optional_bounded_and_non_destructive() -> None:
    source = REFERENCE.read_text()
    assert REFERENCE.stat().st_mode & 0o111
    assert "FREECIV_REFERENCE_REQUIRED" in source
    assert "find \"$root\" -maxdepth 6" in source
    assert "/home/docker/freeciv/share/freeciv/helpdata.txt" in source
    assert "/home/docker/freeciv/share/freeciv/classic" in source
    assert "failed image or checksum verification" in source
    assert 'mv "$staging" "$OUTPUT_DIR"' in source
    # A present but incomplete first prefix must not prevent searching later
    # supported prefixes (for example /usr/share after /home/docker).
    search_block = source.split("for root in /home/docker/freeciv", 1)[1].split("done", 1)[0]
    assert "find \"$root\" -maxdepth 6" in search_block
    assert 'find "$root" -maxdepth 6 -type f -name helpdata.txt -print -quit 2>/dev/null\n      exit 0' not in search_block


def test_reference_extractor_missing_docker_is_nonfatal_unless_required(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text("#!/usr/bin/env bash\nexit 1\n")
    docker.chmod(0o755)
    sg = fake_bin / "sg"
    sg.write_text("#!/usr/bin/env bash\nexit 1\n")
    sg.chmod(0o755)
    env = {"PATH": f"{fake_bin}:/usr/bin:/bin", "FREECIV_REFERENCE_DIR": str(tmp_path / "ref")}
    optional = subprocess.run([str(REFERENCE)], cwd=ROOT, env=env,
                              text=True, capture_output=True, check=False)
    assert optional.returncode == 0
    assert "reference unavailable" in optional.stderr
    required = subprocess.run([str(REFERENCE)], cwd=ROOT,
                              env={**env, "FREECIV_REFERENCE_REQUIRED": "1"},
                              text=True, capture_output=True, check=False)
    assert required.returncode != 0
    assert "reference extraction failed" in required.stderr


def test_reference_extractor_mocked_docker_is_idempotent_and_rejects_tampering(tmp_path: Path) -> None:
    fake_root = tmp_path / "image"
    help_path = fake_root / "usr" / "share" / "freeciv"
    (help_path / "classic").mkdir(parents=True)
    (help_path / "helpdata.txt").write_text("[help_overview]\nname = \\\"Overview\\\"\n")
    (help_path / "classic" / "civ.ruleset").write_text("rules\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "case \"$1\" in\n"
        " info) exit 0;;\n"
        " inspect) if printf '%s' \"$*\" | grep -q '{{.Image}}'; then if [ \"${MISMATCH:-0}\" = 1 ]; then printf 'sha256:wrong\\n'; else printf 'sha256:pinned\\n'; fi; elif printf '%s' \"$*\" | grep -q -- '-f'; then exit 0; fi;;\n"
        " image) if printf '%s' \"$*\" | grep -q '{{.Id}}'; then printf 'sha256:pinned\\n'; else printf '1.0\\n'; fi;;\n"
        " exec) if printf '%s' \"$*\" | grep -q helpdata; then printf '/usr/share/freeciv/helpdata.txt\\n'; else printf '/usr/share/freeciv/classic\\n'; fi;;\n"
        " cp) src=\"${2#*:}\"; src=\"${src%/.}\"; case \"$src\" in */classic) cp -R \"$FAKE_ROOT$src/.\" \"$3\";; *) cp -R \"$FAKE_ROOT$src\" \"$3\";; esac;;\n"
        "esac\n"
    )
    docker.chmod(0o755)
    env = {"PATH": f"{fake_bin}:/usr/bin:/bin", "FAKE_ROOT": str(fake_root),
           "FREECIV_REFERENCE_DIR": str(tmp_path / "reference")}
    first = subprocess.run([str(REFERENCE)], cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    assert first.returncode == 0, first.stderr
    manifest = json.loads((tmp_path / "reference" / "reference.json").read_text())
    assert manifest["helpdata_path"] == "/usr/share/freeciv/helpdata.txt"
    second = subprocess.run([str(REFERENCE)], cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    assert second.returncode == 0
    (tmp_path / "reference" / "helpdata.txt").write_text("tampered")
    tampered = subprocess.run([str(REFERENCE)], cwd=ROOT,
                              env={**env, "FREECIV_REFERENCE_REQUIRED": "1"},
                              text=True, capture_output=True, check=False)
    assert tampered.returncode != 0
    assert "checksum verification" in tampered.stderr
    mismatch = subprocess.run([str(REFERENCE)], cwd=ROOT,
                              env={**env, "MISMATCH": "1", "FREECIV_REFERENCE_REQUIRED": "1"},
                              text=True, capture_output=True, check=False)
    assert mismatch.returncode != 0
    assert "does not use pinned image" in mismatch.stderr


def _write_reference_docker(tmp_path: Path) -> tuple[Path, Path]:
    """Create a Docker CLI mock whose container has only the classic ruleset."""
    fake_root = tmp_path / "image"
    classic = fake_root / "opt" / "freeciv" / "share" / "freeciv" / "classic"
    classic.mkdir(parents=True)
    (classic / "civ.ruleset").write_text("rules-only\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    state = tmp_path / "docker.calls"
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\nset -eu\n"
        "case \"$1\" in\n"
        " info) exit 0;;\n"
        " inspect) if printf '%s' \"$*\" | grep -q '{{.Image}}'; then [ \"${MISMATCH:-0}\" = 1 ] && printf 'sha256:wrong\\n' || printf 'sha256:pinned\\n'; else exit 0; fi;;\n"
        " image) if printf '%s' \"$*\" | grep -q '{{.Id}}'; then printf 'sha256:pinned\\n'; else printf '1.0\\n'; fi;;\n"
        " exec) n=0; [ -f \"$STATE\" ] && n=$(cat \"$STATE\"); n=$((n + 1)); printf '%s' \"$n\" > \"$STATE\"; if [ \"${NO_HELP:-0}\" = 1 ] && [ \"$n\" = 1 ]; then exit 1; fi; if printf '%s' \"$*\" | grep -q 'classic'; then printf '/opt/freeciv/share/freeciv/classic\\n'; else printf '/opt/freeciv/share/freeciv/helpdata.txt\\n'; fi;;\n"
        " cp) src=\"${2#*:}\"; src=\"${src%/.}\"; case \"$src\" in */classic) cp -R \"$FAKE_ROOT$src/.\" \"$3\";; *) cp -R \"$FAKE_ROOT$src\" \"$3\";; esac;;\n"
        "esac\n"
    )
    docker.chmod(0o755)
    return fake_bin, fake_root


def test_reference_extractor_accepts_ruleset_only_and_is_idempotent(tmp_path: Path) -> None:
    fake_bin, fake_root = _write_reference_docker(tmp_path)
    output = tmp_path / "reference"
    env = {"PATH": f"{fake_bin}:/usr/bin:/bin", "FAKE_ROOT": str(fake_root),
           "STATE": str(tmp_path / "docker.calls"), "NO_HELP": "1",
           "FREECIV_REFERENCE_DIR": str(output)}
    first = subprocess.run([str(REFERENCE)], cwd=ROOT, env=env, text=True,
                           capture_output=True, check=False)
    assert first.returncode == 0, first.stderr
    assert "note: bundled helpdata.txt was not found" in first.stderr
    assert "warning: helpdata.txt" not in first.stderr
    manifest = json.loads((output / "reference.json").read_text())
    assert manifest["helpdata_status"] == "unavailable"
    assert manifest["helpdata_path"] is None
    assert manifest["helpdata_sha256"] is None
    assert not (output / "helpdata.txt").exists()
    assert (output / "classic" / "civ.ruleset").read_text() == "rules-only\n"
    before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    second = subprocess.run([str(REFERENCE)], cwd=ROOT, env=env, text=True,
                            capture_output=True, check=False)
    assert second.returncode == 0, second.stderr
    after = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    assert after == before
    (output / "classic" / "civ.ruleset").write_text("tampered\n")
    rejected = subprocess.run([str(REFERENCE)], cwd=ROOT,
                              env={**env, "FREECIV_REFERENCE_REQUIRED": "1"},
                              text=True, capture_output=True, check=False)
    assert rejected.returncode != 0
    assert "checksum verification" in rejected.stderr


def test_reference_extractor_required_help_and_diagnose_errors_are_distinct(tmp_path: Path) -> None:
    fake_bin, fake_root = _write_reference_docker(tmp_path)
    env = {"PATH": f"{fake_bin}:/usr/bin:/bin", "FAKE_ROOT": str(fake_root),
           "STATE": str(tmp_path / "docker.calls"), "NO_HELP": "1",
           "FREECIV_REFERENCE_DIR": str(tmp_path / "required")}
    required = subprocess.run([str(REFERENCE)], cwd=ROOT,
                              env={**env, "FREECIV_REFERENCE_REQUIRE_HELP": "1"},
                              text=True, capture_output=True, check=False)
    assert required.returncode != 0
    assert "helpdata.txt was not found" in required.stderr
    assert not (tmp_path / "required").exists()

    broken = tmp_path / "broken-bin"
    broken.mkdir()
    docker = broken / "docker"
    docker.write_text("#!/usr/bin/env bash\nexit 42\n")
    docker.chmod(0o755)
    diagnostic = subprocess.run([str(REFERENCE), "--diagnose"], cwd=ROOT,
                                env={"PATH": f"{broken}:/usr/bin:/bin",
                                     "FREECIV_REFERENCE_DIR": str(tmp_path / "diag")},
                                text=True, capture_output=True, check=False)
    assert diagnostic.returncode == 0
    assert "Docker API" in diagnostic.stderr
    assert not (tmp_path / "diag").exists()


def test_reference_extractor_rejects_symlinks_in_classic_tree(tmp_path: Path) -> None:
    fake_bin, fake_root = _write_reference_docker(tmp_path)
    classic = fake_root / "opt" / "freeciv" / "share" / "freeciv" / "classic"
    (classic / "escape").symlink_to("/etc/passwd")
    output = tmp_path / "symlink-reference"
    env = {"PATH": f"{fake_bin}:/usr/bin:/bin", "FAKE_ROOT": str(fake_root),
           "STATE": str(tmp_path / "docker.calls"), "NO_HELP": "1",
           "FREECIV_REFERENCE_DIR": str(output), "FREECIV_REFERENCE_REQUIRED": "1"}
    result = subprocess.run([str(REFERENCE)], cwd=ROOT, env=env, text=True,
                            capture_output=True, check=False)
    assert result.returncode != 0
    assert "contains symlinks" in result.stderr
    assert not output.exists()


def test_reference_diagnose_reports_broader_freeciv_paths(tmp_path: Path) -> None:
    fake_bin, fake_root = _write_reference_docker(tmp_path)
    env = {"PATH": f"{fake_bin}:/usr/bin:/bin", "FAKE_ROOT": str(fake_root),
           "STATE": str(tmp_path / "diagnose.calls"),
           "FREECIV_REFERENCE_DIR": str(tmp_path / "diag")}
    result = subprocess.run([str(REFERENCE), "--diagnose"], cwd=ROOT, env=env,
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "Freeciv reference diagnostic" in result.stdout
    assert "image=sha256:pinned" in result.stdout
    assert not (tmp_path / "diag").exists()


def test_launch_checks_fixed_reference_path_when_reusing_healthy_stack() -> None:
    launch = LAUNCH.read_text()
    assert 'artifacts/reference/freeciv/helpdata.txt' in launch
    assert 'extract_freeciv_reference.sh' in launch
    assert 'FREECIV_IMAGE_REF=' in launch
    assert 'Local Freeciv reference missing; attempting extraction' in launch
    assert 'Verifying local Freeciv reference' in launch
    assert 'failed image or checksum verification' in REFERENCE.read_text()


def test_manual_generation_is_wired_into_provision_and_healthy_reuse() -> None:
    launch = LAUNCH.read_text()
    provision = PROVISION.read_text()
    for source in (launch, provision):
        assert "LECHONK_EXTRACT_FREECIV_MANUAL" in source
        assert "FREECIV_MANUAL_REQUIRED" in source
        assert "extract_freeciv_manual.sh" in source
    assert "freeciv-manual" in MANUAL.read_text()
    assert 'artifacts/reference/freeciv-manual' in launch
    assert 'Local Freeciv manual missing; attempting generation' in launch
    assert 'required local Freeciv manual extraction is disabled' in provision
    assert 'required local Freeciv manual extraction is disabled or unavailable' in launch
    # A healthy artifact is handed to the extractor, whose manifest/checksum
    # guard makes repeated launch calls cheap and avoids regeneration.
    assert 'manual_manifest="$manual_dir/reference.json"' in launch


def _make_launch_fixture(
    tmp_path: Path,
    *,
    running: str = "true",
    docker_mode: str = "direct",
) -> tuple[Path, Path]:
    """Build a tiny local git fixture and command mocks for launch.sh."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / ".venv" / "bin").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)

    subrepo = tmp_path / "civrealm"
    subrepo.mkdir()
    subprocess.run(["git", "init", "-q", str(subrepo)], check=True)
    (subrepo / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (subrepo / "value.txt").write_text("base\n")
    subprocess.run(["git", "-C", str(subrepo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(subrepo), "-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base"],
        check=True,
    )
    (subrepo / "value.txt").write_text("patched\n")
    subprocess.run(["git", "-C", str(subrepo), "add", "value.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(subrepo), "-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-qm", "compatibility patch"],
        check=True,
    )
    sha = subprocess.run(["git", "-C", str(subrepo), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
    vendor = root / "vendor" / "civrealm"
    vendor.parent.mkdir()
    vendor.symlink_to(subrepo, target_is_directory=True)
    subprocess.run(["git", "-C", str(root), "update-index", "--add", "--cacheinfo", f"160000,{sha},vendor/civrealm"], check=True)

    launch_copy = root / "scripts" / "launch.sh"
    launch_copy.write_text(LAUNCH.read_text())
    launch_copy.chmod(0o755)
    (root / "scripts" / "provision.sh").write_text("#!/usr/bin/env bash\nprintf provision-called > provision.marker\n")
    (root / "scripts" / "provision.sh").chmod(0o755)
    fake_python = root / ".venv" / "bin" / "python"
    fake_python.write_text("#!/usr/bin/env bash\nprintf python-called > python.marker\n")
    fake_python.chmod(0o755)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "if [ \"${DOCKER_LOG:-}\" ]; then\n"
        "  printf '<%s>\\n' \"$@\" >> \"$DOCKER_LOG\"\n"
        "fi\n"
        "if [ \"${1:-}\" = info ] && [ \"${DOCKER_MODE:-direct}\" = sg ] && [ \"${DOCKER_VIA_SG:-0}\" != 1 ]; then\n"
        "  exit 13\n"
        "fi\n"
        "case \"$*\" in\n"
        "  info*) exit 0;;\n"
        f"  *'{{.State.Running}}'* ) printf '{running}\\n';;\n"
        "  *'{{.Id}}'*) [ \"${DOCKER_MODE:-direct}\" = invalid-image ] && printf 'sha256:wrong\\n' || printf 'sha256:fixture\\n';;\n"
        "  *'{{.Image}}'*) printf 'sha256:fixture\\n';;\n"
        "  *) exit 1;;\n"
        "esac\n"
    )
    (bin_dir / "docker").chmod(0o755)
    (bin_dir / "sg").write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "printf '%s\\n' \"$3\" >> \"$SG_LOG\"\n"
        "DOCKER_VIA_SG=1; export DOCKER_VIA_SG\n"
        "exec bash -c \"$3\"\n"
    )
    (bin_dir / "sg").chmod(0o755)
    (bin_dir / "curl").write_text("#!/usr/bin/env bash\nexit 0\n")
    (bin_dir / "curl").chmod(0o755)
    return root, bin_dir


def _run_launch_fixture(root: Path, bin_dir: Path, *, docker_mode: str,
                        sg_log: Path, docker_log: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(root / "scripts" / "launch.sh"), "--turns", "1"],
        cwd=root,
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "DOCKER_MODE": docker_mode,
            "SG_LOG": str(sg_log),
            "DOCKER_LOG": str(docker_log),
        },
        text=True,
        capture_output=True,
        check=False,
    )


def test_launch_health_checks_use_direct_docker_without_sg(tmp_path: Path) -> None:
    root, bin_dir = _make_launch_fixture(tmp_path, docker_mode="direct")
    sg_log = tmp_path / "sg.log"
    docker_log = tmp_path / "docker.log"
    result = _run_launch_fixture(root, bin_dir, docker_mode="direct",
                                 sg_log=sg_log, docker_log=docker_log)
    assert result.returncode == 0, result.stderr
    assert not sg_log.exists()
    assert not (root / "provision.marker").exists()


def test_launch_health_checks_fall_back_to_sg_after_permission_denied(tmp_path: Path) -> None:
    root, bin_dir = _make_launch_fixture(tmp_path, docker_mode="sg")
    sg_log = tmp_path / "sg.log"
    docker_log = tmp_path / "docker.log"
    result = _run_launch_fixture(root, bin_dir, docker_mode="sg",
                                 sg_log=sg_log, docker_log=docker_log)
    assert result.returncode == 0, result.stderr
    assert not (root / "provision.marker").exists()
    sg_calls = sg_log.read_text().splitlines()
    assert sg_calls
    assert all(call.startswith("docker ") for call in sg_calls)
    assert any(r"\{\{.State.Running\}\}" in call for call in sg_calls)


@pytest.mark.parametrize("docker_mode", ["invalid-image", "daemon-down"])
def test_launch_health_check_provisions_on_genuine_docker_failure(
        tmp_path: Path, docker_mode: str) -> None:
    root, bin_dir = _make_launch_fixture(tmp_path, docker_mode=docker_mode)
    if docker_mode == "daemon-down":
        (bin_dir / "docker").write_text(
            "#!/usr/bin/env bash\n"
            "exit 19\n"
        )
        (bin_dir / "docker").chmod(0o755)
        # The group fallback is unavailable too, so this is a real health
        # failure rather than a permission-mode fallback.
        (bin_dir / "sg").write_text("#!/usr/bin/env bash\nexit 20\n")
        (bin_dir / "sg").chmod(0o755)
    result = _run_launch_fixture(
        root, bin_dir, docker_mode=docker_mode,
        sg_log=tmp_path / "sg.log", docker_log=tmp_path / "docker.log")
    assert result.returncode == 0, result.stderr
    assert (root / "provision.marker").exists()


def test_launch_sg_fallback_preserves_shell_quoting_for_docker_arguments(tmp_path: Path) -> None:
    root, bin_dir = _make_launch_fixture(tmp_path, docker_mode="sg")
    sg_log = tmp_path / "sg.log"
    result = _run_launch_fixture(root, bin_dir, docker_mode="sg",
                                 sg_log=sg_log, docker_log=tmp_path / "docker.log")
    assert result.returncode == 0, result.stderr
    calls = sg_log.read_text().splitlines()
    assert any(r"\{\{.State.Running\}\}" in call for call in calls)
    assert any(r"\{\{.Id\}\}" in call for call in calls)


def test_launch_dock_sg_fallback_preserves_adversarial_argument_boundaries(
        tmp_path: Path) -> None:
    """Execute the launch helper with shell-hostile arguments through fake sg."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker-args.log"
    sg_log = tmp_path / "sg-command.log"
    marker = tmp_path / "command-substitution-ran"

    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "if [ \"${DOCKER_VIA_SG:-0}\" != 1 ]; then exit 13; fi\n"
        "for arg in \"$@\"; do printf '<%q>\\n' \"$arg\" >> \"$DOCKER_LOG\"; done\n"
    )
    docker.chmod(0o755)
    sg = fake_bin / "sg"
    sg.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "printf '%s' \"$3\" > \"$SG_LOG\"\n"
        "DOCKER_VIA_SG=1; export DOCKER_VIA_SG\n"
        "exec bash -c \"$3\"\n"
    )
    sg.chmod(0o755)

    # Keep this test coupled to the production helper rather than maintaining
    # a second hand-written copy of its quoting logic.
    launch = LAUNCH.read_text()
    helper = launch.split("dock() {", 1)[1].split("\n}\n", 1)[0]
    probe = tmp_path / "probe.sh"
    adversarial_command = (
        "dock inspect 'space value' 'quote\"value' "
        + "'$(touch " + str(marker) + ")' "
        + "'semi;colon' $'line\\nvalue'\n"
    )
    probe.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        f"dock() {{{helper}\n}}\n"
        + adversarial_command
    )
    probe.chmod(0o755)
    result = subprocess.run(
        [str(probe)],
        cwd=ROOT,
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "DOCKER_LOG": str(docker_log),
            "SG_LOG": str(sg_log),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not marker.exists()
    logged = docker_log.read_text().splitlines()
    assert logged[0] == "<inspect>"
    assert any("space\\ value" in item for item in logged)
    assert any('quote\\"value' in item for item in logged)
    assert any("\\$\\(touch" in item and str(marker) in item for item in logged)
    assert any("semi\\;colon" in item for item in logged)
    assert any("line\\nvalue" in item for item in logged)
    assert len(logged) == 6


@pytest.mark.parametrize(("running", "expect_provision"), [("true", False), ("false", True)])
def test_launch_fast_path_is_executable_and_fail_closed(tmp_path: Path, running: str, expect_provision: bool) -> None:
    root, bin_dir = _make_launch_fixture(tmp_path, running=running)
    result = subprocess.run(
        [str(root / "scripts" / "launch.sh"), "--turns", "1"],
        cwd=root,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (root / "provision.marker").exists() is expect_provision


def _manual_docker_mock(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_root = tmp_path / "container"
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\nset -eu\n"
        "case \"$1\" in\n"
        " info) exit 0;;\n"
        " inspect) if printf '%s' \"$*\" | grep -q '{{.Image}}'; then [ \"${MISMATCH:-0}\" = 1 ] && printf 'sha256:wrong\\n' || printf 'sha256:pinned\\n'; else exit 0; fi;;\n"
        " image) if printf '%s' \"$*\" | grep -q '{{.Id}}'; then printf 'sha256:pinned\\n'; else printf '3.2.91-dev\\n'; fi;;\n"
        " exec)\n"
        "   if printf '%s' \"$*\" | grep -q 'rm -rf'; then rm -rf \"$FAKE_ROOT/tmp/lechonk-freeciv-manual.mock\";\n"
        "   elif printf '%s' \"$*\" | grep -q -- '--version'; then printf 'Freeciv version 3.2.91-dev\\n';\n"
        "   elif printf '%s' \"$*\" | grep -q 'for p in'; then printf '%s\\n' \"${MANUAL_PATH:-/home/docker/freeciv/bin/freeciv-manual}\";\n"
        "   elif printf '%s' \"$*\" | grep -q -- '--ruleset classic --wiki'; then\n"
        "     d=/tmp/lechonk-freeciv-manual.mock; mkdir -p \"$FAKE_ROOT$d\"; for n in 1 2 3 4 5 6 7 8; do printf 'manual %s\\n' \"$n\" > \"$FAKE_ROOT$d/classic$n.mediawiki\"; done; printf '%s\\n' \"$d\";\n"
        "   fi;;\n"
        " cp) src=\"${2#*:}\"; src=\"${src%/.}\"; cp -R \"$FAKE_ROOT$src/.\" \"$3\";;\n"
        "esac\n"
    )
    docker.chmod(0o755)
    return fake_bin, {"PATH": f"{fake_bin}:/usr/bin:/bin", "FAKE_ROOT": str(fake_root)}


def test_manual_extractor_mock_is_idempotent_and_rejects_tampering(tmp_path: Path) -> None:
    fake_bin, env = _manual_docker_mock(tmp_path)
    output = tmp_path / "manual"
    env = {**env, "FREECIV_MANUAL_DIR": str(output)}
    first = subprocess.run([str(MANUAL)], cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    assert first.returncode == 0, first.stderr
    manifest = json.loads((output / "reference.json").read_text())
    assert manifest["ruleset"] == "classic"
    assert set(manifest["files"]) == {f"classic{i}.mediawiki" for i in range(1, 9)}
    assert not (tmp_path / "container" / "tmp" / "lechonk-freeciv-manual.mock").exists()
    before = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
    second = subprocess.run([str(MANUAL)], cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    assert second.returncode == 0
    assert {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()} == before
    (output / "classic1.mediawiki").write_text("tampered")
    rejected = subprocess.run([str(MANUAL)], cwd=ROOT,
                              env={**env, "FREECIV_MANUAL_REQUIRED": "1"},
                              text=True, capture_output=True, check=False)
    assert rejected.returncode != 0
    assert "checksum verification" in rejected.stderr


def test_manual_extractor_uses_a_nondefault_discovered_binary_path(tmp_path: Path) -> None:
    _, env = _manual_docker_mock(tmp_path)
    output = tmp_path / "manual"
    result = subprocess.run(
        [str(MANUAL)], cwd=ROOT,
        env={**env, "MANUAL_PATH": "/usr/bin/freeciv-manual",
             "FREECIV_MANUAL_DIR": str(output)},
        text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads((output / "reference.json").read_text())["manual_binary"] == "/usr/bin/freeciv-manual"


def test_manual_extractor_is_optional_and_verifies_pinned_image(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text("#!/usr/bin/env bash\n[ \"$1\" = info ] && exit 1\nexit 1\n")
    docker.chmod(0o755)
    env = {"PATH": f"{fake_bin}:/usr/bin:/bin", "FREECIV_MANUAL_DIR": str(tmp_path / "out")}
    optional = subprocess.run([str(MANUAL)], cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    assert optional.returncode == 0 and "manual unavailable" in optional.stderr
    required = subprocess.run([str(MANUAL)], cwd=ROOT,
                              env={**env, "FREECIV_MANUAL_REQUIRED": "1"},
                              text=True, capture_output=True, check=False)
    assert required.returncode != 0 and "manual extraction failed" in required.stderr


def test_manual_extractor_rejects_container_image_mismatch(tmp_path: Path) -> None:
    _, env = _manual_docker_mock(tmp_path)
    result = subprocess.run(
        [str(MANUAL)], cwd=ROOT,
        env={**env, "MISMATCH": "1", "FREECIV_MANUAL_REQUIRED": "1",
             "FREECIV_MANUAL_DIR": str(tmp_path / "out")},
        text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert "does not use pinned image" in result.stderr


def test_manual_extractor_source_has_safe_temp_cleanup_and_no_network() -> None:
    source = MANUAL.read_text()
    assert MANUAL.stat().st_mode & 0o111
    assert "mktemp -d /tmp/lechonk-freeciv-manual.XXXXXX" in source
    assert "rm -rf -- \"$container_dir\"" in source
    assert "--ruleset classic --wiki" in source
    assert "curl" not in source and "wget" not in source
