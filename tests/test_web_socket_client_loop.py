# SPDX-License-Identifier: MIT
"""Regression tests for CivRealm's websocket loop ownership boundary."""

import asyncio
import threading

import pytest

from civrealm.freeciv.connectivity.web_socket_client import WebSocketClient


def _client(loop):
    client = WebSocketClient()
    class OwnedIOLoop:
        asyncio_loop = loop
        def __init__(self):
            self.callbacks = []
        def add_callback(self, callback):
            self.callbacks.append(callback)
            loop.call_soon_threadsafe(callback)
        def start(self):
            loop.run_until_complete(asyncio.sleep(0))
        def stop(self):
            pass
    client._io_loop = OwnedIOLoop()
    return client


def test_socket_loop_restores_missing_thread_local_loop():
    loop = asyncio.new_event_loop()
    previous = getattr(asyncio.get_event_loop_policy()._local, "_loop", None)
    try:
        client = _client(loop)
        asyncio.set_event_loop(None)
        client._ensure_owned_asyncio_loop("send")
        assert asyncio.get_event_loop() is loop
    finally:
        asyncio.set_event_loop(previous)
        loop.close()


def test_socket_loop_rejects_different_loop_before_send():
    owned = asyncio.new_event_loop()
    other = asyncio.new_event_loop()
    previous = getattr(asyncio.get_event_loop_policy()._local, "_loop", None)
    try:
        client = _client(owned)
        asyncio.set_event_loop(other)
        with pytest.raises(RuntimeError, match="loop changed"):
            client._ensure_owned_asyncio_loop("send")
    finally:
        asyncio.set_event_loop(previous)
        owned.close()
        other.close()


def test_socket_loop_diagnostic_contains_only_bounded_state():
    owned = asyncio.new_event_loop()
    other = asyncio.new_event_loop()
    previous = getattr(asyncio.get_event_loop_policy()._local, "_loop", None)
    try:
        client = _client(owned)
        diagnostic = client._loop_diagnostic(owned, other)
        assert diagnostic["owned"]["id"].startswith("0x")
        assert diagnostic["current"]["id"].startswith("0x")
        assert set(diagnostic["owned"]) == {"id", "closed", "running"}
        assert "message" not in diagnostic
    finally:
        asyncio.set_event_loop(previous)
        owned.close()
        other.close()


def test_socket_close_uses_owned_loop(monkeypatch):
    loop = asyncio.new_event_loop()
    previous = getattr(asyncio.get_event_loop_policy()._local, "_loop", None)
    try:
        client = _client(loop)
        class Socket:
            def __init__(self):
                self.closed = False
            def close(self):
                self.closed = True
        socket = Socket()
        client._ws_connection = socket
        client._on_connection_close = lambda: None
        calls = []
        client._io_loop.stop = lambda: calls.append("stop")
        client._io_loop.start = lambda: calls.append("start")
        asyncio.set_event_loop(None)
        client.close()
        assert socket.closed is True
        assert calls == ["stop", "start"]
    finally:
        asyncio.set_event_loop(previous)
        loop.close()


def test_socket_send_restores_loop_and_sends_once():
    loop = asyncio.new_event_loop()
    previous = getattr(asyncio.get_event_loop_policy()._local, "_loop", None)
    try:
        client = _client(loop)
        class Socket:
            def __init__(self):
                self.calls = []
            def write_message(self, value):
                self.calls.append(value)
        socket = Socket()
        client._ws_connection = socket
        asyncio.set_event_loop(None)
        client.send({"kind": "mutation"})
        assert socket.calls == []
        loop.run_until_complete(asyncio.sleep(0))
        assert len(socket.calls) == 1
        assert asyncio.get_event_loop() is loop
    finally:
        asyncio.set_event_loop(previous)
        loop.close()


def test_socket_send_then_start_restores_same_loop_without_resend():
    loop = asyncio.new_event_loop()
    previous = getattr(asyncio.get_event_loop_policy()._local, "_loop", None)
    try:
        client = _client(loop)
        class Socket:
            def __init__(self):
                self.calls = []
            def write_message(self, value):
                self.calls.append(value)
        socket = Socket()
        client._ws_connection = socket
        starts = []
        client._io_loop.start = lambda: (starts.append(asyncio.get_event_loop()),
                                         loop.run_until_complete(asyncio.sleep(0)))
        client._io_loop.stop = lambda: None
        asyncio.set_event_loop(loop)
        client.send({"kind": "mutation"})
        assert len(socket.calls) == 0
        # Tornado/CivRealm can clear the thread-local slot between send and
        # its response loop.  Starting the loop must restore the same owner,
        # never replaying the mutation.
        asyncio.set_event_loop(None)
        client.start_ioloop()
        assert starts == [loop]
        assert len(socket.calls) == 1
    finally:
        asyncio.set_event_loop(previous)
        loop.close()


def test_socket_queues_actual_write_on_owned_running_loop():
    loop = asyncio.new_event_loop()
    previous = getattr(asyncio.get_event_loop_policy()._local, "_loop", None)
    try:
        client = _client(loop)
        class Socket:
            def __init__(self):
                self.calls = []
            def write_message(self, value):
                assert asyncio.get_running_loop() is loop
                self.calls.append(value)
        socket = Socket()
        client._ws_connection = socket
        client._io_loop.start = loop.run_forever
        client._io_loop.stop = loop.stop
        client.send({"n": 1})
        client.send({"n": 2})
        loop.run_until_complete(asyncio.sleep(0))
        assert len(socket.calls) == 2
    finally:
        asyncio.set_event_loop(previous)
        loop.close()


def test_socket_deferred_write_failure_is_propagated_without_retry():
    loop = asyncio.new_event_loop()
    previous = getattr(asyncio.get_event_loop_policy()._local, "_loop", None)
    try:
        client = _client(loop)
        class Socket:
            def __init__(self):
                self.calls = 0
            def write_message(self, value):
                assert asyncio.get_running_loop() is loop
                self.calls += 1
                raise OSError("write broke")
        socket = Socket()
        client._ws_connection = socket
        client._io_loop.start = loop.run_forever
        client._io_loop.stop = loop.stop
        client.send({"n": 1})
        with pytest.raises(RuntimeError, match="deferred websocket write failed"):
            client.start_ioloop()
        assert socket.calls == 1
        with pytest.raises(RuntimeError, match="deferred websocket write failed"):
            client.send({"n": 2})
        assert socket.calls == 1
    finally:
        asyncio.set_event_loop(previous)
        loop.close()


def test_socket_other_thread_schedules_on_owner_loop():
    loop = asyncio.new_event_loop()
    previous = getattr(asyncio.get_event_loop_policy()._local, "_loop", None)
    try:
        client = _client(loop)
        class Socket:
            def __init__(self):
                self.calls = []
            def write_message(self, value):
                assert asyncio.get_running_loop() is loop
                self.calls.append(value)
        socket = Socket()
        client._ws_connection = socket
        error = []
        def send_from_other_thread():
            try:
                client.send({"n": 1})
            except BaseException as exc:
                error.append(exc)
        thread = threading.Thread(target=send_from_other_thread)
        thread.start()
        thread.join()
        loop.run_until_complete(asyncio.sleep(0))
        assert error == []
        assert len(socket.calls) == 1
    finally:
        asyncio.set_event_loop(previous)
        loop.close()


def test_socket_close_abandons_queued_writes():
    loop = asyncio.new_event_loop()
    previous = getattr(asyncio.get_event_loop_policy()._local, "_loop", None)
    try:
        client = _client(loop)
        class Socket:
            def __init__(self):
                self.calls = []
                self.closed = False
            def write_message(self, value):
                self.calls.append(value)
            def close(self):
                self.closed = True
        socket = Socket()
        client._ws_connection = socket
        client._on_connection_close = lambda: None
        client.send({"n": 1})
        client.close()
        assert socket.closed
        assert socket.calls == []
    finally:
        asyncio.set_event_loop(previous)
        loop.close()


def test_socket_immediate_future_failure_clears_remaining_queue():
    loop = asyncio.new_event_loop()
    previous = getattr(asyncio.get_event_loop_policy()._local, "_loop", None)
    try:
        client = _client(loop)
        class Socket:
            def __init__(self):
                self.calls = []
            def write_message(self, value):
                self.calls.append(value)
                future = loop.create_future()
                future.set_exception(OSError("future broke"))
                return future
        socket = Socket()
        client._ws_connection = socket
        client._io_loop.start = loop.run_forever
        client._io_loop.stop = loop.stop
        client.send({"n": 1})
        client.send({"n": 2})
        with pytest.raises(RuntimeError, match="deferred websocket write failed"):
            client.start_ioloop()
        assert len(socket.calls) == 1
    finally:
        asyncio.set_event_loop(previous)
        loop.close()
