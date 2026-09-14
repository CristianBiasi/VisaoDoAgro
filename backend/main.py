from fastapi import FastAPI

from backend.api.routes.detection import router as detection_router
from backend.api.routes.alerts import router as alerts_router
from backend.api.routes.stream import router as stream_router
from backend.api.routes.metrics import router as metrics_router

app = FastAPI(
    title="VisaoDoAgro API",
    description="Backend para detecção de obstáculos em drones agrícolas.",
    version="0.1.0",
)

app.include_router(detection_router)
app.include_router(stream_router)
app.include_router(alerts_router)
app.include_router(metrics_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
