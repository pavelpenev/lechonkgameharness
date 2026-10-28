"""Small, side-effect-limited helpers for run-artifact lifecycle guards."""

from __future__ import annotations

import os
import secrets
from pathlib import Path


class ArtifactCollisionError(RuntimeError):
    """A fresh attempt would overwrite existing user artifacts."""


_RETRY_ARTIFACTS = (
    "conversation.jsonl", "state.json", "api_calls.jsonl",
    "model_traces.jsonl", "transcript.jsonl",
    "token_budget_table.json", "token_budget_table.md", "metrics_report.json",
    "metrics_report.md", "manifest_runtime.json", "run_meta.json",
    "recovery.json", "infra_failure_attempt1.json",
    "ruleset_reference.json", "ruleset_reference_meta.json",
)


def claim_run_directory(path: str | Path, *, attempt: int, resuming: bool,
                        token: str | None = None) -> str:
    """Validate the run root and claim it with an exclusive marker."""
    directory = Path(path)
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise ArtifactCollisionError(f"run directory is not a regular directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    if attempt <= 1 and not resuming:
        ensure_fresh_run_directory(directory, attempt=attempt, resuming=resuming)
    marker = directory / ".run-owner"
    if marker.is_symlink() or (marker.exists() and not marker.is_file()):
        raise ArtifactCollisionError(f"invalid run ownership marker: {marker}")
    if attempt > 1 and not marker.exists():
        raise ArtifactCollisionError(f"retry run directory is unclaimed: {directory}")
    if attempt > 1:
        expected = marker.read_text(encoding="utf-8").strip()
        if not token or not secrets.compare_digest(expected, token):
            raise ArtifactCollisionError(f"retry ownership token mismatch: {directory}")
        return expected
    if not marker.exists():
        token = token or secrets.token_hex(32)
        try:
            fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(token + "\n")
        except FileExistsError as exc:
            raise ArtifactCollisionError(f"run directory is already claimed: {directory}") from exc
        return token
    raise ArtifactCollisionError(f"run directory is already claimed: {directory}")


def prepare_retry_artifacts(path: str | Path, *, attempt: int) -> None:
    """Move all known retry outputs, including checkpoint directories."""
    if attempt <= 1:
        return
    root = Path(path)
    names = (*_RETRY_ARTIFACTS, "checkpoints")
    existing: list[tuple[Path, Path]] = []
    # Preflight every source and destination before moving anything.
    for name in names:
        artifact = root / name
        if not artifact.exists() and not artifact.is_symlink():
            continue
        backup = Path(str(artifact) + ".attempt1")
        if backup.exists() or backup.is_symlink():
            raise ArtifactCollisionError(f"refusing to overwrite prior retry artifact: {backup}")
        if artifact.is_symlink():
            raise ArtifactCollisionError(f"invalid retry artifact path: {artifact}")
        expected_dir = name == "checkpoints"
        if artifact.is_dir() != expected_dir:
            raise ArtifactCollisionError(f"invalid retry artifact type: {artifact}")
        if not expected_dir and not artifact.is_file():
            raise ArtifactCollisionError(f"invalid retry artifact path: {artifact}")
        existing.append((artifact, backup))
    for artifact, backup in existing:
        os.replace(artifact, backup)


def ensure_fresh_run_directory(path: str | Path, *, attempt: int,
                               resuming: bool) -> None:
    """Allow empty provisioning directories, but reject nonempty fresh runs."""
    directory = Path(path)
    if attempt <= 1 and not resuming and directory.is_dir():
        if any(directory.iterdir()):
            raise ArtifactCollisionError(
                f"fresh run rejected: target run directory already contains "
                f"artifacts: {directory}")


def prepare_attempt_artifact(path: str | Path, *, attempt: int) -> Path:
    """Rotate an artifact for retry, never replacing an existing backup."""
    artifact = Path(path)
    if not artifact.exists():
        return artifact
    if attempt <= 1:
        artifact.unlink()
        return artifact
    backup = Path(str(artifact) + ".attempt1")
    if backup.exists():
        raise ArtifactCollisionError(
            f"refusing to overwrite prior retry artifact: {backup}")
    os.replace(artifact, backup)
    return artifact
