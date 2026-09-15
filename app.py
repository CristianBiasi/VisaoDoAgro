import asyncio
import io
import json
import socket
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import psutil
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from config import ROOT, settings
from detector import YoloDetector
from models import Event, SettingsPatch, Telemetry

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
    from av import VideoFrame
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

app = FastAPI(title="AgroSafe Vision", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")

@app.middleware("http")
async def disable_frontend_cache(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response

state = {
    "phone_connected": False, "camera_connected": False, "stream_connected": False,
    "gps_active": False, "sensors_active": False, "mission_started": time.time(),
    "telemetry": Telemetry(), "detections": [], "events": deque(maxlen=100),
    "ai_fps": 0.0, "inference_ms": 0.0, "video_fps": 0.0, "resolution": "UNAVAILABLE",
    "device": "UNAVAILABLE", "alert_count": 0, "pcs": set(), "last_alert": {}, "video_clients": set(),
}
detector = YoloDetector()
SECURE = (ROOT / ".certs" / "cert.pem").exists() and (ROOT / ".certs" / "key.pem").exists()
SCHEME = "https" if SECURE else "http"


def local_ip():
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80)); return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def add_event(message, level="INFO", track_id=None):
    event = Event(message=message, level=level, track_id=track_id)
    state["events"].appendleft(event.model_dump())
    if level in {"HIGH", "CRITICAL"}:
        state["alert_count"] += 1


def snapshot():
    telemetry = state["telemetry"].model_dump()
    primary = max(state["detections"], key=lambda item: (item["risk"], item["proximity"]), default=None)
    return {"status": {key: state[key] for key in ("phone_connected", "camera_connected", "stream_connected", "gps_active", "sensors_active")}, "telemetry": telemetry, "detections": state["detections"], "primary_threat": primary, "events": list(state["events"]), "metrics": {"ai_fps": round(state["ai_fps"], 1), "inference_ms": round(state["inference_ms"], 1), "video_fps": round(state["video_fps"], 1), "resolution": state["resolution"], "device": state["device"], "objects": len(state["detections"]), "alerts": state["alert_count"], "mission_time": int(time.time() - state["mission_started"])}}


class ProcessedTrack(VideoStreamTrack):
    def __init__(self, source):
        super().__init__(); self.source = source; self.last = time.perf_counter(); self.frames = 0; self.last_ai = 0.0; self.last_broadcast = 0.0; self.cached = None

    async def recv(self):
        frame = await self.source.recv()
        image = frame.to_ndarray(format="bgr24")
        state["phone_connected"] = state["camera_connected"] = state["stream_connected"] = True
        state["resolution"] = f"{image.shape[1]}x{image.shape[0]}"
        now = time.perf_counter(); self.frames += 1
        elapsed = now - self.last
        if elapsed >= 1:
            state["video_fps"] = self.frames / elapsed; self.frames = 0; self.last = now
        now = time.perf_counter()
        should_infer = self.cached is None or now - self.last_ai >= 1 / settings.inference_fps
        if should_infer:
            try:
                inference_image = image
                max_dimension = max(inference_image.shape[:2])
                if max_dimension > 640:
                    scale = 640 / max_dimension
                    inference_image = cv2.resize(inference_image, (int(inference_image.shape[1] * scale), int(inference_image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
                self.cached = detector.process(inference_image)
                self.last_ai = now
                state["detections"] = [item.model_dump() for item in detector.last_detections]
                state["inference_ms"] = detector.last_inference; state["ai_fps"] = 1 / max(0.001, detector.last_inference / 1000); state["device"] = detector.device
                for item in state["detections"]:
                    if item["risk"] in {"MEDIUM", "HIGH", "CRITICAL"}:
                        key = f"{item['track_id']}:{item['risk']}"
                        if state["last_alert"].get(item["track_id"]) != key:
                            state["last_alert"][item["track_id"]] = key; add_event(f"{item['object_class'].upper()} risk {item['risk']}", item["risk"], item["track_id"])
            except Exception as exc:
                add_event(f"AI unavailable: {type(exc).__name__}", "WARNING")
                self.cached = image
        processed = self.cached
        output = VideoFrame.from_ndarray(processed, format="bgr24"); output.pts = frame.pts; output.time_base = frame.time_base
        success, encoded = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if success and state["video_clients"] and now - self.last_broadcast >= 1 / 8:
            await asyncio.gather(*(client.send_bytes(encoded.tobytes()) for client in tuple(state["video_clients"])), return_exceptions=True)
            self.last_broadcast = now
        return output


@app.get("/")
async def dashboard():
    return FileResponse(ROOT / "frontend" / "index.html")

@app.get("/phone")
async def phone():
    return FileResponse(ROOT / "frontend" / "phone.html")

@app.get("/api/session")
async def session():
    return {"session": "AF-4281", "local_ip": local_ip(), "phone_url": f"{SCHEME}://{local_ip()}:8000/phone", "webrtc": WEBRTC_AVAILABLE}

@app.get("/api/qr")
async def qr():
    import qrcode
    image = qrcode.make(f"{SCHEME}://{local_ip()}:8000/phone"); output = io.BytesIO(); image.save(output, format="PNG"); output.seek(0)
    return StreamingResponse(output, media_type="image/png")

@app.get("/api/state")
async def get_state():
    return snapshot()

@app.get("/api/settings")
async def get_settings():
    return settings.model_dump()

@app.patch("/api/settings")
async def patch_settings(payload: SettingsPatch):
    global settings
    try:
        settings = settings.model_copy(update=payload.values)
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": json.loads(exc.json())})
    add_event("Settings updated", "INFO")
    return settings.model_dump()

@app.post("/api/telemetry")
async def telemetry(payload: Telemetry):
    state["telemetry"] = payload; state["gps_active"] = payload.latitude is not None; state["sensors_active"] = any(value is not None for value in (payload.heading, payload.pitch, payload.roll)); return {"ok": True}

@app.post("/api/mission/start")
async def start_mission():
    state["mission_started"] = time.time(); state["alert_count"] = 0; state["events"].clear(); add_event("Mission started", "INFO"); return {"ok": True}

@app.post("/api/mission/stop")
async def stop_mission():
    add_event("Mission stopped", "INFO"); return {"ok": True}

@app.post("/api/offer")
async def offer(request: Request):
    if not WEBRTC_AVAILABLE: return JSONResponse(status_code=503, content={"detail": "aiortc is not installed"})
    params = await request.json(); pc = RTCPeerConnection(); state["pcs"].add(pc)
    @pc.on("track")
    def on_track(track):
        if track.kind == "video": pc.addTrack(ProcessedTrack(track))
    @pc.on("connectionstatechange")
    async def on_state_change():
        if pc.connectionState in {"failed", "closed", "disconnected"}: state["stream_connected"] = False; state["phone_connected"] = False; await pc.close(); state["pcs"].discard(pc)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=params["sdp"], type=params["type"]))
    answer = await pc.createAnswer(); await pc.setLocalDescription(answer)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

@app.websocket("/ws/events")
async def events(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(snapshot()); await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass

@app.websocket("/ws/video")
async def video(websocket: WebSocket):
    await websocket.accept()
    state["video_clients"].add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        state["video_clients"].discard(websocket)

@app.get("/api/health")
async def health():
    return {"ok": True, "cpu": psutil.cpu_percent(), "ram": psutil.virtual_memory().percent}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
