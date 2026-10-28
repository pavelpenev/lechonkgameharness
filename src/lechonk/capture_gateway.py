"""Fail-open, append-only capture gateway for the interactive runner.

The gateway deliberately keeps capture I/O off the runner/model callback
thread. Controls have a separate bounded queue with one reserved terminal
slot; both queues are published by their reserved sequence so the append-only
file remains deterministic. This module does not interpret stdout or model
text.
"""

from __future__ import annotations

import copy
import json
import os
import queue
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVENT_VERSION = "capture-event-v1"
MANIFEST_VERSION = "capture-manifest-v1"
_ABNORMAL_STATUSES = frozenset({
    "process_exit", "cancelled", "canceled", "exception", "error",
    "failed", "truncated", "interrupted",
})


def _wall_time() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _copy_payload(value: Any) -> Any:
    """Copy callback data without changing model text or nested structures."""
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


class CaptureGateway:
    """Bounded asynchronous capture writer.

    ``offer_trace`` and ``offer_control`` are nonblocking and safe to call
    from model/network callback threads.  All public methods fail open: a
    capture failure is reflected in ``manifest.json`` where possible and
    never raised into gameplay.
    """

    def __init__(self, capture_dir: str | os.PathLike[str], *, run_id: str,
                 expected_username: str, child_pid: int | None = None,
                 trace_capacity: int = 4096, control_capacity: int = 512):
        # Resolve before the runner can change CWD (AdapterSession does this
        # for its isolated workdir). Relative manifest paths otherwise keep
        # writing events through an open FD while atomic updates fail in the
        # new directory.
        self.capture_dir = Path(capture_dir).expanduser().resolve()
        self.run_id = str(run_id)
        self.capture_id = uuid.uuid4().hex
        self.expected_username = expected_username
        self.child_pid = int(child_pid if child_pid is not None else os.getpid())
        self._trace = queue.Queue(maxsize=trace_capacity)
        # One extra slot is reserved for capture_finished. Ordinary control
        # records can never consume the terminal reservation.
        self._control_capacity = control_capacity
        self._control = queue.Queue(maxsize=control_capacity + 1)
        self._stop = threading.Event()
        self._closed = False
        self._lock = threading.RLock()
        self._queue_lock = threading.RLock()
        # Atomic replace prevents torn JSON, but concurrent replace writers
        # can still publish an older snapshot after finalization. Serialize
        # the complete temp-write/fsync/replace transaction as well.
        self._manifest_write_lock = threading.Lock()
        self._seq = 0
        self._dropped_trace = 0
        self._dropped_control = 0
        self._write_errors = 0
        self._last_error: str | None = None
        self._manifest: dict[str, Any] = {
            "version": MANIFEST_VERSION,
            "schema_version": MANIFEST_VERSION,
            "capture_id": self.capture_id,
            "run_id": self.run_id,
            "child_pid": self.child_pid,
            # Presenter binding uses this exact field; it is derived from the
            # runner's trusted FREECIV_USERNAME, not parsed from stdout.
            "expected_username": expected_username,
            "expected_freeciv_username": expected_username,
            "session_port": None,
            "session_ready": False,
            "status": "starting",
            "partial": False,
            "complete": False,
            "identity_assurance": "provisional_username_witness",
            "production_ready": False,
            "nonce_evidence": "not_provided",
            "capture_quality": "degraded_prototype",
            "binding_mode": "prototype_binding",
            "capture_events": "capture-events.jsonl",
            "created_wall_time": _wall_time(),
            "created_monotonic_ns": time.monotonic_ns(),
            "dropped_trace": 0,
            "dropped_control": 0,
            "write_errors": 0,
        }
        self.events_path: Path | None = None
        self.manifest_path: Path | None = None
        self._thread: threading.Thread | None = None
        self._terminal_enqueued = False
        self._finishing = False

    @classmethod
    def create(cls, capture_dir: str | os.PathLike[str], **kwargs: Any) -> "CaptureGateway":
        """Create a unique capture directory and start its writer.

        ``exist_ok=False`` is intentional: an existing directory is never
        overwritten, even when a caller reuses a run id.
        """
        gateway = cls(capture_dir, **kwargs)
        try:
            gateway.capture_dir.mkdir(parents=True, exist_ok=False)
            os.chmod(gateway.capture_dir, 0o700)
            gateway.events_path = gateway.capture_dir / "capture-events.jsonl"
            gateway.manifest_path = gateway.capture_dir / "manifest.json"
            gateway._write_manifest()
            gateway._thread = threading.Thread(
                target=gateway._writer, name="capture-writer", daemon=True)
            gateway._thread.start()
            gateway.offer_control("capture_started", {
                "capture_id": gateway.capture_id,
                "run_id": gateway.run_id,
            })
        except FileExistsError:
            # Never silently downgrade an explicit no-overwrite violation.
            raise
        except Exception as exc:
            gateway._note_error(exc)
        return gateway

    def _note_error(self, exc: BaseException) -> None:
        message = f"capture gateway error: {type(exc).__name__}: {exc}"
        with self._lock:
            self._write_errors += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._manifest["partial"] = True
            self._manifest["write_errors"] = self._write_errors
            self._manifest["last_error"] = self._last_error
        # Capture failures are fail-open for gameplay, but never invisible to
        # the operator. Avoid raising from callback/writer threads.
        try:
            print(f"[lechonk-capture] {message}", file=sys.stderr, flush=True)
        except Exception:
            pass

    def _write_manifest(self) -> None:
        if self.manifest_path is None:
            return
        with self._manifest_write_lock:
            with self._lock:
                data = dict(self._manifest)
                data["dropped_trace"] = self._dropped_trace
                data["dropped_control"] = self._dropped_control
                data["write_errors"] = self._write_errors
                if self._last_error:
                    data["last_error"] = self._last_error
            tmp = self.manifest_path.with_name(f".{self.manifest_path.name}.{uuid.uuid4().hex}.tmp")
            try:
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2, sort_keys=True, ensure_ascii=False,
                              default=str)
                    fh.write("\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                os.chmod(tmp, 0o600)
                os.replace(tmp, self.manifest_path)
                try:
                    dir_fd = os.open(self.capture_dir, os.O_RDONLY)
                    try:
                        os.fsync(dir_fd)
                    finally:
                        os.close(dir_fd)
                except OSError:
                    pass
            except Exception as exc:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                self._note_error(exc)

    def _event(self, kind: str, payload: Any) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            seq = self._seq
        return {
            "version": EVENT_VERSION,
            "schema_version": EVENT_VERSION,
            "capture_id": self.capture_id,
            "run_id": self.run_id,
            "capture_seq": seq,
            "monotonic_ns": time.monotonic_ns(),
            "wall_time": _wall_time(),
            "kind": str(kind),
            "event_kind": str(kind),
            "payload": _copy_payload(payload),
        }

    def _offer_receipt(self, kind: str, payload: Any, target: queue.Queue,
                       control: bool) -> int | None:
        """Enqueue one event and return its assigned sequence when accepted.

        Sequence assignment and queue insertion are kept under the same lock so
        a caller using the receipt can refer to the exact event it caused.  A
        rejected event consumes no sequence number.
        """
        with self._queue_lock:
            if self._closed:
                return None
            try:
                # Leave one control slot for the terminal marker.
                if control and kind != "capture_finished":
                    with target.mutex:
                        full_reserved = len(target.queue) >= self._control_capacity
                    if full_reserved:
                        raise queue.Full
                event = self._event(kind, payload)
                target.put_nowait(event)
                if kind == "capture_finished":
                    self._terminal_enqueued = True
                return int(event["capture_seq"])
            except queue.Full:
                with self._lock:
                    if control:
                        self._dropped_control += 1
                    else:
                        self._dropped_trace += 1
                    self._manifest["partial"] = True
                return None
            except Exception as exc:
                self._note_error(exc)
                return None

    def _offer(self, kind: str, payload: Any, target: queue.Queue, control: bool) -> bool:
        return self._offer_receipt(kind, payload, target, control) is not None

    def offer_trace(self, kind: str, payload: Any) -> bool:
        return self._offer(kind, payload, self._trace, False)

    def offer_control(self, kind: str, payload: Any) -> bool:
        return self._offer(kind, payload, self._control, True)

    def offer_control_receipt(self, kind: str, payload: Any) -> int | None:
        """Offer a control event and return its exact ``capture_seq``.

        ``None`` means the event was rejected because the gateway is closed,
        the bounded queue is full, or capture I/O failed.  This is intended for
        optional recording checkpoints; ordinary gameplay should continue to
        use the boolean, fail-open ``offer_control`` API.
        """
        return self._offer_receipt(kind, payload, self._control, True)

    def session_ready(self, event: dict[str, Any]) -> bool:
        port = event.get("port")
        with self._lock:
            self._manifest["session_ready"] = True
            self._manifest["session_port"] = port
            self._manifest["session_ready_wall_time"] = _wall_time()
            # Keep the manifest small; immutable ruleset/help blobs belong in
            # their existing runner artifacts, not in every atomic rewrite.
            self._manifest["session_ready_metadata"] = {
                key: _copy_payload(value) for key, value in event.items()
                if key not in {"ruleset_reference_json", "help_reference_json",
                               "classic_reference_json", "manual_reference_json"}
            }
        self._write_manifest()
        return self.offer_control("session_ready", event)

    def ledger_result(self, payload: Any) -> bool:
        return self.offer_control("ledger_result", payload)

    def finish(self, *, status: str = "finished", partial: bool | None = None,
               payload: Any = None) -> None:
        with self._lock:
            if self._closed or self._finishing:
                return
            self._finishing = True
        # The terminal marker must not leapfrog already-accepted trace
        # records merely because controls have priority in the writer.
        drain_deadline = time.monotonic() + 5.0
        while not self._trace.empty() and time.monotonic() < drain_deadline:
            time.sleep(0.001)
        terminal_ok = self.offer_control(
            "capture_finished", payload or {"status": status})
        self._stop.set()
        writer_joined = True
        if self._thread is not None:
            self._thread.join(timeout=10)
            writer_joined = not self._thread.is_alive()
        with self._lock:
            self._closed = True
            self._manifest["status"] = status
            incomplete = bool(partial if partial is not None else self._manifest.get("partial"))
            # A process-exit/abort marker is not evidence of a clean capture.
            # A normal earlier finish is idempotent and returns above, so this
            # cannot downgrade an already-complete capture.
            incomplete = incomplete or str(status).lower() in _ABNORMAL_STATUSES
            incomplete = incomplete or not terminal_ok or not writer_joined or self._write_errors > 0
            self._manifest["complete"] = not incomplete
            self._manifest["partial"] = incomplete
            if not terminal_ok:
                self._manifest["finalization_error"] = "capture_finished event was not enqueued"
            elif not writer_joined:
                self._manifest["finalization_error"] = "capture writer did not stop before timeout"
            self._manifest["finished_wall_time"] = _wall_time()
            self._manifest["finished_monotonic_ns"] = time.monotonic_ns()
        self._write_manifest()

    close = finish

    def _writer(self) -> None:
        if self.events_path is None:
            return
        fh = None
        try:
            fh = open(self.events_path, "a", encoding="utf-8")
            os.chmod(self.events_path, 0o600)
            while not self._stop.is_set() or not self._control.empty() or not self._trace.empty():
                event = None
                with self._queue_lock:
                    # Controls remain separate and bounded, but publication
                    # follows reservation order so presenters never see a
                    # lower capture_seq after a higher one.
                    control_head = self._control.queue[0] if self._control.queue else None
                    trace_head = self._trace.queue[0] if self._trace.queue else None
                    if control_head is None and trace_head is None:
                        pass
                    elif trace_head is None or (
                            control_head is not None and
                            control_head["capture_seq"] < trace_head["capture_seq"]):
                        event = self._control.get_nowait()
                    else:
                        event = self._trace.get_nowait()
                if event is None:
                    time.sleep(0.01)
                    continue
                fh.write(json.dumps(event, sort_keys=True, ensure_ascii=False, default=str) + "\n")
                fh.flush()
            os.fsync(fh.fileno())
        except Exception as exc:
            self._note_error(exc)
        finally:
            if fh is not None:
                try:
                    fh.close()
                except OSError:
                    pass
            self._write_manifest()
