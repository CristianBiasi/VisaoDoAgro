from fastapi import FastAPI

from backend.api.routes.detection import router as detection_router

app = FastAPI(
    title="VisaoDoAgro API",
    description="Backend para detecção de obstáculos em drones agrícolas.",
    version="0.1.0",
)

app.include_router(detection_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
