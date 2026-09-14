from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.api.dependencies import get_stream_processing_service
from backend.infrastructure.video_source import VideoSource, VideoSourceError
from backend.services.stream_processing_service import StreamProcessingService


router = APIRouter(prefix="/detect/stream", tags=["stream"])


class StreamStartRequest(BaseModel):
    video_path: str = Field(min_length=1)


@router.post("/start")
def start_stream(
    request: StreamStartRequest,
    stream_processing_service: StreamProcessingService = Depends(
        get_stream_processing_service
    ),
) -> dict[str, float | int]:
    try:
        summary = stream_processing_service.process(VideoSource(request.video_path))
    except VideoSourceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return summary.to_dict()
