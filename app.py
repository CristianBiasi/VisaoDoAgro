import asyncio
from datetime import datetime, timezone
import io
import json
import logging
import socket
import tempfile
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import psutil
from fastapi import FastAPI, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

import config
from config import ROOT, updated_settings, PERFORMANCE_PROFILES, PROXIMITY_PRESETS
from detector import YoloDetector
from models import Event, SettingsPatch, Telemetry
from risk import Risk
from database import Base, engine, SessionLocal, Missao, EventoAlerta, TelemetriaPonto, ObstaculoDetectado, Configuracao
from local_video import LocalVideoProcessor

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from video_pipeline import LatestFrame, ProcessedTrack, MAX_OVERLAY_AGE
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

logger = logging.getLogger(__name__)
settings = config.settings
detector = YoloDetector()
tracks = {}
local_video_processor: LocalVideoProcessor | None = None
state = {
    "phone_connected": False, "camera_connected": False, "stream_connected": False,
    "gps_active": False, "sensors_active": False, "mission_started": time.time(),
    "mission_id": None, "session_code": "AF-4281",
    "telemetry": Telemetry(), "detections": [], "events": deque(maxlen=100),
    "ai_fps": 0.0, "inference_ms": 0.0, "video_fps": 0.0, "resolution": "UNAVAILABLE",
    "device": "UNAVAILABLE", "alert_count": 0, "pcs": set(), "last_alert": {},
    "video_clients": set(), "metrics": {},
}
SECURE = (ROOT / ".certs" / "cert.pem").exists() and (ROOT / ".certs" / "key.pem").exists()
SCHEME = "https" if SECURE else "http"


@asynccontextmanager
async def lifespan(application):
    # Inicializa as tabelas do Banco de Dados
    Base.metadata.create_all(bind=engine)
    
    # Carrega configuracoes do BD se existirem
    with SessionLocal() as db:
        conf = db.query(Configuracao).filter(Configuracao.id == 1).first()
        if conf:
            try:
                config.settings = config.Settings(**conf.settings_json)
                global settings
                settings = config.settings
            except Exception as e:
                logger.error("Erro ao carregar configuracoes do BD: %s", e)
        else:
            # Salva o default
            new_conf = Configuracao(id=1, settings_json=settings.model_dump())
            db.add(new_conf)
            db.commit()

    yield
    await asyncio.gather(*(close_peer(pc) for pc in tuple(state["pcs"])))
    for slot in tuple(state["video_clients"]):
        slot.close()


app = FastAPI(title="AgroSafe Vision", version="1.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")


@app.middleware("http")
async def disable_frontend_cache(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response


def local_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("8.8.8.8", 80))
            return probe.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def add_event(message, level="INFO", track_id=None):
    event_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state["events"].appendleft(Event(timestamp=event_timestamp, message=message, level=level, track_id=track_id).model_dump())
    if level in {"HIGH", "CRITICAL"}:
        state["alert_count"] += 1
        
    if state["mission_id"]:
        try:
            with SessionLocal() as db:
                novo_alerta = EventoAlerta(
                    missao_id=state["mission_id"],
                    timestamp=event_timestamp,
                    nivel=level,
                    mensagem=message,
                    track_id=track_id,
                    lat=state["telemetry"].latitude,
                    lon=state["telemetry"].longitude,
                    alt=state["telemetry"].altitude
                )
                db.add(novo_alerta)
                
                # Atualizar total de alertas na missao se for o caso
                if level in {"HIGH", "CRITICAL"}:
                    missao = db.query(Missao).filter(Missao.id == state["mission_id"]).first()
                    if missao:
                        missao.alertas = state["alert_count"]
                db.commit()
        except Exception as e:
            logger.error("Erro ao salvar evento no BD: %s", e)


def publish_result(result, metrics):
    state["phone_connected"] = state["camera_connected"] = state["stream_connected"] = True
    state["resolution"] = f"{result.width}x{result.height}"
    state["detections"] = [item.model_dump() for item in result.detections]
    state["metrics"] = metrics
    now = time.perf_counter()
    visible = {item.track_id for item in result.detections}
    state["last_alert"] = {key: value for key, value in state["last_alert"].items() if key in visible}
    
    db_items = []
    
    for item in result.detections:
        if item.risk not in {"MEDIUM", "HIGH", "CRITICAL"}:
            continue
        previous_level, previous_time = state["last_alert"].get(item.track_id, (None, 0))
        if item.risk != previous_level or now - previous_time >= settings.alert_cooldown:
            add_event(f"{item.object_class.upper()} risk {item.risk}", item.risk, item.track_id)
            state["last_alert"][item.track_id] = (item.risk, now)
            
        # Para obstáculos confirmados (alto risco), salvar/atualizar no BD
        if item.risk in {"HIGH", "CRITICAL"} and state["mission_id"]:
            db_items.append(item)

    if db_items and state["mission_id"]:
        try:
            with SessionLocal() as db:
                for item in db_items:
                    # Tenta encontrar se o obstáculo já foi mapeado nesta missão
                    obs = db.query(ObstaculoDetectado).filter(
                        ObstaculoDetectado.missao_id == state["mission_id"],
                        ObstaculoDetectado.track_id == item.track_id
                    ).first()
                    
                    if not obs:
                        obs = ObstaculoDetectado(
                            missao_id=state["mission_id"],
                            track_id=item.track_id,
                            classe=item.object_class,
                            risco_max=item.risk,
                            confianca_max=item.confidence,
                            proximidade_min=item.proximity,
                            lat=state["telemetry"].latitude,
                            lon=state["telemetry"].longitude,
                            primeira_deteccao=time.time(),
                            ultima_deteccao=time.time()
                        )
                        db.add(obs)
                    else:
                        # Atualiza valores se piorar o risco ou ficar mais perto/confiável
                        if Risk[item.risk].value > Risk[obs.risco_max].value:
                            obs.risco_max = item.risk
                        if item.confidence > obs.confianca_max:
                            obs.confianca_max = item.confidence
                        if item.proximity < obs.proximidade_min:
                            obs.proximidade_min = item.proximity
                        obs.ultima_deteccao = time.time()
                db.commit()
        except Exception as e:
            logger.error("Erro ao salvar obstáculos no BD: %s", e)


def publish_error(exc):
    state["detections"] = []
    add_event(f"AI unavailable: {type(exc).__name__}: {exc}", "WARNING")


def snapshot():
    metrics = dict(state["metrics"])
    detections = state["detections"]
    active_track = next(iter(tracks.values()), None)
    if active_track:
        live = active_track.metrics()
        # Keep the measured completion latency; result age continues to increase.
        live.pop("server_detection_ms", None)
        metrics.update(live)
        result = active_track.latest_result
        if (not result or active_track.result_settings is not settings
                or time.perf_counter() - result.received_at > MAX_OVERLAY_AGE):
            detections = []
    primary = max(detections, key=lambda item: (Risk[item["risk"]], item["proximity"]), default=None)
    defaults = {"ai_fps": 0, "inference_ms": 0, "video_fps": 0, "device": "UNAVAILABLE",
                "server_detection_ms": None, "result_age_ms": None, "dropped_frames": 0,
                "backend": "Ultralytics / PyTorch"}
    return {
        "status": {key: state[key] for key in (
            "phone_connected", "camera_connected", "stream_connected", "gps_active", "sensors_active",
        )},
        "telemetry": state["telemetry"].model_dump(),
        "detections": detections, "primary_threat": primary, "events": list(state["events"]),
        "metrics": {**defaults, **metrics, "resolution": state["resolution"],
                    "objects": len(detections), "alerts": state["alert_count"],
                    "mission_time": int(time.time() - state["mission_started"])},
    }


@app.get("/")
async def dashboard():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/phone")
async def phone():
    return FileResponse(ROOT / "frontend" / "phone.html")


@app.get("/api/session")
async def session():
    address = local_ip()
    return {"session": "AF-4281", "local_ip": address,
            "phone_url": f"{SCHEME}://{address}:8000/phone", "webrtc": WEBRTC_AVAILABLE}


@app.get("/api/qr")
async def qr():
    import qrcode
    image = qrcode.make(f"{SCHEME}://{local_ip()}:8000/phone")
    output = io.BytesIO()
    image.save(output, format="PNG")
    output.seek(0)
    return StreamingResponse(output, media_type="image/png")


@app.get("/api/state")
async def get_state():
    return snapshot()


@app.get("/api/settings")
async def get_settings():
    return settings.model_dump()


@app.get("/api/presets")
async def presets():
    return {"performance_profile": PERFORMANCE_PROFILES, "proximity_preset": PROXIMITY_PRESETS}


@app.get("/api/model")
async def model_info():
    return {"configured_model": settings.model_path, "loaded_model": detector.loaded_model,
            "available_classes": list(detector.available_classes.values()),
            "missing_classes": detector.missing_classes, "device": detector.device,
            "pending_reload": detector.needs_reload(settings)}


@app.patch("/api/settings")
async def patch_settings(payload: SettingsPatch):
    global settings
    try:
        candidate = updated_settings(settings, payload.values)
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": json.loads(exc.json())})
    # No mutation of a configuration currently used by the inference worker.
    settings = candidate
    config.settings = candidate
    state["detections"] = []
    
    # Salvar configuracoes no BD
    try:
        with SessionLocal() as db:
            conf = db.query(Configuracao).filter(Configuracao.id == 1).first()
            if conf:
                conf.settings_json = settings.model_dump()
            else:
                conf = Configuracao(id=1, settings_json=settings.model_dump())
                db.add(conf)
            db.commit()
    except Exception as e:
        logger.error("Erro ao salvar configuracoes no BD: %s", e)
        
    add_event("Settings updated", "INFO")
    return settings.model_dump()


@app.post("/api/telemetry")
async def telemetry(payload: Telemetry):
    state["telemetry"] = payload
    state["gps_active"] = payload.latitude is not None
    state["sensors_active"] = any(value is not None for value in (payload.heading, payload.pitch, payload.roll))
    
    if state["mission_id"]:
        try:
            with SessionLocal() as db:
                ponto = TelemetriaPonto(
                    missao_id=state["mission_id"],
                    timestamp=time.time(),
                    lat=payload.latitude,
                    lon=payload.longitude,
                    alt=payload.altitude,
                    velocidade=payload.speed_kmh,
                    heading=payload.heading,
                    pitch=payload.pitch,
                    roll=payload.roll
                )
                db.add(ponto)
                db.commit()
        except Exception as e:
            logger.error("Erro ao salvar telemetria no BD: %s", e)
            
    return {"ok": True}


@app.post("/api/mission/start")
async def start_mission():
    state["mission_started"] = time.time()
    state["alert_count"] = 0
    state["events"].clear()
    state["last_alert"].clear()
    
    # Criar registro de missao no BD
    try:
        with SessionLocal() as db:
            nova_missao = Missao(
                sessao=state["session_code"],
                inicio=state["mission_started"],
                modelo=settings.model_path,
                status="EM_ANDAMENTO"
            )
            db.add(nova_missao)
            db.commit()
            db.refresh(nova_missao)
            state["mission_id"] = nova_missao.id
    except Exception as e:
        logger.error("Erro ao criar missao no BD: %s", e)
        
    add_event("Mission started", "INFO")
    return {"ok": True}


@app.post("/api/mission/stop")
async def stop_mission():
    # Atualizar registro no BD
    if state["mission_id"]:
        try:
            with SessionLocal() as db:
                missao = db.query(Missao).filter(Missao.id == state["mission_id"]).first()
                if missao:
                    missao.fim = time.time()
                    missao.duracao = missao.fim - missao.inicio
                    missao.status = "CONCLUIDA"
                    missao.alertas = state["alert_count"]
                    db.commit()
            state["mission_id"] = None
        except Exception as e:
            logger.error("Erro ao encerrar missao no BD: %s", e)
            
    add_event("Mission stopped", "INFO")
    return {"ok": True}


async def close_peer(pc):
    track = tracks.get(pc)
    if track:
        await track.aclose()
    tracks.pop(pc, None)
    state["pcs"].discard(pc)
    if pc.connectionState != "closed":
        await pc.close()
    if not state["pcs"]:
        state["stream_connected"] = state["phone_connected"] = state["camera_connected"] = False
        state["detections"] = []
        state["metrics"] = {}


@app.post("/api/camera/stop")
async def stop_camera():
    global local_video_processor
    if local_video_processor and local_video_processor.running:
        await local_video_processor.stop()
    await asyncio.gather(*(close_peer(pc) for pc in tuple(state["pcs"])))
    state["stream_connected"] = state["phone_connected"] = state["camera_connected"] = False
    state["detections"] = []
    state["metrics"] = {}
    return {"ok": True}


@app.post("/api/video/upload")
async def upload_video(file: UploadFile):
    """Accept a local video file and process it server-side with OpenCV + YOLO."""
    global local_video_processor
    # Stop any existing stream first
    if state["pcs"]:
        await asyncio.gather(*(close_peer(pc) for pc in tuple(state["pcs"])))
    if local_video_processor and local_video_processor.running:
        await local_video_processor.stop()

    # Save uploaded file to a temp location
    temp_dir = ROOT / ".tmp_videos"
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / f"upload_{int(time.time())}.mp4"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Create processor and start background task
    local_video_processor = LocalVideoProcessor(
        detector, lambda: settings, publish_result, publish_error, state["video_clients"]
    )
    local_video_processor.start(str(temp_path))
    add_event("Local video started", "INFO")
    return {"ok": True, "filename": file.filename}


@app.post("/api/video/stop")
async def stop_video():
    """Stop local video processing."""
    global local_video_processor
    if local_video_processor and local_video_processor.running:
        await local_video_processor.stop()
    state["stream_connected"] = state["phone_connected"] = state["camera_connected"] = False
    state["detections"] = []
    state["metrics"] = {}
    add_event("Local video stopped", "INFO")
    return {"ok": True}


@app.post("/api/offer")
async def offer(request: Request):
    if not WEBRTC_AVAILABLE:
        return JSONResponse(status_code=503, content={"detail": "aiortc is not installed"})
    params = await request.json()
    # One mission/camera: prevent mixing track IDs and model state from different streams.
    if state["pcs"]:
        return JSONResponse(status_code=409, content={"detail": "End the current camera stream before connecting another"})
    if params.get("type") != "offer" or not isinstance(params.get("sdp"), str):
        return JSONResponse(status_code=422, content={"detail": "A WebRTC SDP offer is required"})
    pc = RTCPeerConnection()
    state["pcs"].add(pc)

    @pc.on("track")
    def on_track(track):
        if track.kind == "video" and pc not in tracks:
            processed = ProcessedTrack(track, detector, lambda: settings,
                                       publish_result, publish_error, state["video_clients"])
            tracks[pc] = processed
            pc.addTrack(processed)
            state["phone_connected"] = state["camera_connected"] = state["stream_connected"] = True

    @pc.on("connectionstatechange")
    async def on_state_change():
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            await close_peer(pc)

    try:
        await pc.setRemoteDescription(RTCSessionDescription(sdp=params["sdp"], type=params["type"]))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    except Exception:
        logger.exception("WebRTC negotiation failed")
        await close_peer(pc)
        return JSONResponse(status_code=400, content={"detail": "WebRTC negotiation failed"})


@app.websocket("/ws/events")
async def events(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(snapshot())
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/video")
async def video(websocket: WebSocket):
    await websocket.accept()
    slot = LatestFrame()
    state["video_clients"].add(slot)

    async def send_frames():
        while True:
            await websocket.send_bytes(await slot.get())

    async def receive_disconnect():
        while True:
            await websocket.receive_text()

    sender = asyncio.create_task(send_frames())
    receiver = asyncio.create_task(receive_disconnect())
    try:
        await asyncio.wait([sender, receiver], return_when=asyncio.FIRST_COMPLETED)
    finally:
        state["video_clients"].discard(slot)
        slot.close()
        sender.cancel()
        receiver.cancel()
        await asyncio.gather(sender, receiver, return_exceptions=True)


@app.get("/api/health")
async def health():
    return {"ok": True, "cpu": psutil.cpu_percent(), "ram": psutil.virtual_memory().percent}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
