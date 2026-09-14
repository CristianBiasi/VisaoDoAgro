from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from backend.api.dependencies import get_notifier
from backend.infrastructure.notifier import Notifier


router = APIRouter(tags=["alerts"])


@router.websocket("/ws/alerts")
async def alerts_websocket(
    websocket: WebSocket,
    notifier: Notifier = Depends(get_notifier),
) -> None:
    await notifier.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await notifier.disconnect(websocket)
