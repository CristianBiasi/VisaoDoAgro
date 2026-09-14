import asyncio
import json
import logging
import threading
from typing import Any

from fastapi import WebSocket

from backend.domain.entities import Alert


logger = logging.getLogger(__name__)


class Notifier:
    """Publishes alerts to connected WebSockets and to structured logs."""

    def __init__(self) -> None:
        self._connections: dict[int, tuple[WebSocket, asyncio.AbstractEventLoop]] = {}
        self._connections_lock = threading.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        with self._connections_lock:
            self._connections[id(websocket)] = (
                websocket,
                asyncio.get_running_loop(),
            )

    async def disconnect(self, websocket: WebSocket) -> None:
        with self._connections_lock:
            self._connections.pop(id(websocket), None)

    def publish(self, alert: Alert) -> None:
        payload = {
            "event": "alert",
            "alert": alert.model_dump(mode="json"),
        }
        self._log_alert(payload)

        with self._connections_lock:
            connections = list(self._connections.values())

        for websocket, loop in connections:
            self._schedule_send(websocket, loop, payload)

    @staticmethod
    def _log_alert(payload: dict[str, Any]) -> None:
        logger.info(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    def _schedule_send(
        self,
        websocket: WebSocket,
        loop: asyncio.AbstractEventLoop,
        payload: dict[str, Any],
    ) -> None:
        coroutine = self._send(websocket, payload)
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop is loop:
            loop.create_task(coroutine)
            return

        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        future.add_done_callback(self._consume_send_result)

    async def _send(self, websocket: WebSocket, payload: dict[str, Any]) -> None:
        try:
            await websocket.send_json(payload)
        except Exception:
            logger.exception("Could not publish alert to WebSocket client")
            await self.disconnect(websocket)

    @staticmethod
    def _consume_send_result(future: Any) -> None:
        try:
            future.result()
        except Exception:
            logger.exception("Could not schedule alert notification")
