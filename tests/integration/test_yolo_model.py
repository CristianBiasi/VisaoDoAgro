import numpy as np

from backend.domain.entities import Detection, ObstacleClass
from backend.infrastructure.yolo_model import YoloModel


class FakeTensor:
    def __init__(self, values: list) -> None:
        self.values = values

    def cpu(self) -> "FakeTensor":
        return self

    def tolist(self) -> list:
        return self.values


class FakeBoxes:
    xyxy = FakeTensor([[10.0, 20.0, 100.0, 200.0]])
    conf = FakeTensor([0.87])
    cls = FakeTensor([14])


class FakeResult:
    boxes = FakeBoxes()


class FakeYoloModel:
    def __call__(self, frame: np.ndarray, verbose: bool = False) -> list[FakeResult]:
        assert frame.shape == (64, 64, 3)
        assert verbose is False
        return [FakeResult()]


def test_detect_generated_image_returns_domain_detections(monkeypatch) -> None:
    def fake_yolo(model_path: str) -> FakeYoloModel:
        assert model_path == "fake-model.pt"
        return FakeYoloModel()

    monkeypatch.setattr("backend.infrastructure.yolo_model.YOLO", fake_yolo)
    model = YoloModel(
        model_path="fake-model.pt",
        class_mapping={14: ObstacleClass.AVE},
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    detections = model.detect(image)

    assert isinstance(detections, list)
    assert len(detections) == 1
    assert all(isinstance(detection, Detection) for detection in detections)
    assert detections[0].obstacle_class is ObstacleClass.AVE
    assert detections[0].confidence == 0.87
