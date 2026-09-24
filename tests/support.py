"""Fixtures compartilhadas: nenhum teste deve acessar o banco configurado no .env."""
import importlib
import os
from collections import deque
from unittest.mock import MagicMock, patch

from config import Settings
from models import Telemetry


def load_database():
    # create_engine não conecta ao importar. SQLite dispensa credenciais e driver PG.
    with patch.dict(os.environ, {"BD": "sqlite:///:memory:"}), patch("dotenv.load_dotenv"):
        return importlib.import_module("database")


def load_app():
    load_database()
    return importlib.import_module("app")


class IsolatedAppMixin:
    """Restaura globals e mocks mesmo se um teste falhar; não inicia o lifespan."""

    def setUp(self):
        super().setUp()
        self.app = load_app()
        settings = Settings()
        state = {
            **self.app.state,
            "phone_connected": False, "camera_connected": False, "stream_connected": False,
            "gps_active": False, "sensors_active": False, "mission_started": 0,
            "mission_id": None, "telemetry": Telemetry(), "detections": [],
            "events": deque(maxlen=100), "alert_count": 0, "last_alert": {},
            "pcs": set(), "video_clients": set(), "metrics": {}, "resolution": "UNAVAILABLE",
        }
        self.db = MagicMock(name="session")
        self.db.query.return_value.filter.return_value.first.return_value = None
        self.session_factory = MagicMock(name="SessionLocal")
        self.session_factory.return_value.__enter__.return_value = self.db
        for target, name, value in (
            (self.app, "state", state), (self.app, "tracks", {}),
            (self.app, "settings", settings), (self.app.config, "settings", settings),
            (self.app, "SessionLocal", self.session_factory),
            (self.app, "local_video_processor", None),
        ):
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
