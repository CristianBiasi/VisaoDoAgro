from fastapi import FastAPI

app = FastAPI(
    title="VisaoDoAgro API",
    description="Backend para detecção de obstáculos em drones agrícolas.",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
