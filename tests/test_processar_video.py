"""Testes unitários e de integração para processar_video.py."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

# Garante que Ultralytics não tente acessar AppData restrito no sandbox
_cfg = Path(__file__).resolve().parent.parent / ".ultralytics"
(_cfg / "Ultralytics").mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(_cfg)

from processar_video import (
    calculate_queue_size,
    draw_detections,
    frames_to_emit,
    get_class_color,
    get_class_label,
    get_ffmpeg_executable,
    load_model,
    main,
    process_video,
    select_device,
)


def create_synthetic_video(path: Path, frame_count: int = 50, fps: float = 25.0, width: int = 160, height: int = 120) -> Path:
    """Cria um vídeo sintético simples para testes rápidos."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(frame_count):
        # Frame colorido com gradiente temporal
        frame = np.full((height, width, 3), (i * 5) % 256, dtype=np.uint8)
        # Linha em movimento para dar conteúdo visual
        cv2.line(frame, (0, (i * 2) % height), (width, (i * 2) % height), (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    return path


class FakeBoxes:
    """Mock leve do objeto boxes do YOLO."""

    def __init__(self, data):
        self._data = np.array(data, dtype=np.float32)

    def __len__(self):
        return len(self._data)

    @property
    def data(self):
        class CpuTensor:
            def __init__(self, arr):
                self._arr = arr

            def cpu(self):
                return self

            def numpy(self):
                return self._arr

        return CpuTensor(self._data)


class FakeResult:
    """Mock de resultado individual de inferência YOLO."""

    def __init__(self, boxes_data):
        self.boxes = FakeBoxes(boxes_data)


class FakeYOLO:
    """Mock da classe YOLO para testes desacoplados de GPU e pesos reais."""

    def __init__(self, *args, **kwargs):
        self.names = {0: "cable", 1: "tower_lattice", 2: "tower_tucohy", 3: "tower_wooden"}

    def predict(self, frames, *args, **kwargs):
        # Retorna uma detecção de torre treliçada para cada frame
        return [FakeResult([[20, 20, 80, 80, 0.90, 1]]) for _ in frames]

    def track(self, frame, *args, **kwargs):
        # Suporte a chamada de tracking
        return [FakeResult([[20, 20, 80, 80, 1, 0.90, 1]])]


class ProcessarVideoTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_frames_to_emit_various_fps(self):
        """1. frames_to_emit mantém a duração correta em 30 FPS, 60 FPS e 25 FPS."""
        duration_s = 10.0
        out_fps = 30.0

        for src_fps in [30.0, 60.0, 25.0]:
            total_src_frames = int(duration_s * src_fps)
            total_emitted = 0
            next_out_ts = 0.0

            for i in range(total_src_frames):
                src_ts = i / src_fps
                count, next_out_ts = frames_to_emit(src_ts, next_out_ts, src_fps, out_fps)
                total_emitted += count

            expected_out_frames = int(duration_s * out_fps)
            # A duração do vídeo de saída deve ficar a ±1 frame da esperada
            self.assertLessEqual(
                abs(total_emitted - expected_out_frames),
                1,
                f"Falha para src_fps={src_fps}: emitidos {total_emitted}, esperado ~{expected_out_frames}",
            )

            # Relações específicas
            if src_fps == 30.0:
                self.assertEqual(total_emitted, total_src_frames)
            elif src_fps == 60.0:
                # Com 60 FPS, emite cerca de metade dos frames
                self.assertAlmostEqual(total_emitted / total_src_frames, 0.5, delta=0.02)
            elif src_fps == 25.0:
                # Com 25 FPS, emite cerca de 1.2x
                self.assertAlmostEqual(total_emitted / total_src_frames, 1.2, delta=0.02)

    def test_draw_detections(self):
        """2. draw_detections preserva shape/dtype, usa rótulo PT-BR e suporta classes desconhecidas."""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        class_names = {0: "cable", 1: "tower_lattice"}

        # Caso A: Sem detecções
        out_empty = draw_detections(frame, [], class_names=class_names)
        self.assertEqual(out_empty.shape, frame.shape)
        self.assertEqual(out_empty.dtype, frame.dtype)

        # Caso B: Detecção de classe conhecida em PT-BR
        dets = [[10, 10, 100, 100, 0.85, 1]]  # tower_lattice -> Torre treliçada
        hud_info = {"frame_idx": 1, "total_frames": 100, "timestamp_s": 0.5, "counts": {"Torre treliçada": 1}}
        out_det = draw_detections(frame, dets, class_names=class_names, hud_info=hud_info)
        self.assertEqual(out_det.shape, frame.shape)
        self.assertEqual(out_det.dtype, np.uint8)
        # Garante que houve desenho (pixels diferentes de zero)
        self.assertGreater(np.sum(out_det), 0)

        # Labels esperados
        self.assertEqual(get_class_label("tower_lattice"), "Torre treliçada")
        self.assertEqual(get_class_label("cable"), "Cabo")
        self.assertEqual(get_class_label("tower_tucohy"), "Torre tubular/concreto")
        self.assertEqual(get_class_label("tower_wooden"), "Torre de madeira")

        # Caso C: Classe desconhecida
        unknown_dets = [[10, 10, 50, 50, 0.70, 99]]
        unknown_names = {99: "objeto_estranho"}
        out_unknown = draw_detections(frame, unknown_dets, class_names=unknown_names)
        self.assertEqual(out_unknown.shape, frame.shape)
        self.assertGreater(np.sum(out_unknown), 0)

    def test_select_device(self):
        """3. select_device prefere CUDA quando disponível e cai para CPU caso contrário."""
        with patch("torch.cuda.is_available", return_value=False):
            dev, half = select_device("auto")
            self.assertEqual((dev, half), ("cpu", False))

            dev, half = select_device("cuda")
            self.assertEqual((dev, half), ("cpu", False))

        with patch("torch.cuda.is_available", return_value=True):
            dev, half = select_device("auto")
            self.assertEqual((dev, half), ("cuda:0", True))

            dev, half = select_device("cuda")
            self.assertEqual((dev, half), ("cuda:0", True))

            dev, half = select_device("cpu")
            self.assertEqual((dev, half), ("cpu", False))

    def test_calculate_queue_size(self):
        """4. calculate_queue_size respeita os limites mínimo (8) e máximo (64)."""
        # Resoluções comuns
        q_720p = calculate_queue_size(1280, 720, max_ram_gb=10.0)
        self.assertTrue(8 <= q_720p <= 64)

        q_1080p = calculate_queue_size(1920, 1080, max_ram_gb=10.0)
        self.assertTrue(8 <= q_1080p <= 64)

        q_4k = calculate_queue_size(3840, 2160, max_ram_gb=10.0)
        self.assertTrue(8 <= q_4k <= 64)

        # Limite inferior em situação extrema de memória
        q_min = calculate_queue_size(3840, 2160, max_ram_gb=0.001)
        self.assertEqual(q_min, 8)

        # Limite superior com memória abundante e resolução baixa
        q_max = calculate_queue_size(160, 120, max_ram_gb=100.0)
        self.assertEqual(q_max, 64)

    @patch("processar_video.load_model", return_value=FakeYOLO())
    def test_end_to_end_cli(self, mock_load):
        """5. Integração ponta a ponta via CLI: processa vídeo sintético e gera MP4 + JSON."""
        src_video = self.dir_path / "entrada_25fps.mp4"
        dst_video = self.dir_path / "saida_30fps.mp4"
        create_synthetic_video(src_video, frame_count=50, fps=25.0, width=160, height=120)

        # Executa main via argumentos CLI
        ret = main([
            "--input", str(src_video),
            "--output", str(dst_video),
            "--device", "cpu",
            "--no-audio",
            "--batch", "2",
        ])

        self.assertEqual(ret, 0)
        self.assertTrue(dst_video.exists(), "Vídeo de saída não foi gerado.")

        # Valida propriedades do vídeo final
        cap = cv2.VideoCapture(str(dst_video))
        self.assertTrue(cap.isOpened())
        out_fps = cap.get(cv2.CAP_PROP_FPS)
        out_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self.assertAlmostEqual(out_fps, 30.0, delta=1.0)
        # 50 frames a 25 fps = 2.0 segundos -> em 30 fps esperamos cerca de 60 frames
        self.assertAlmostEqual(out_frames, 60, delta=2)

        # Valida relatório JSON
        json_path = dst_video.with_suffix(".json")
        self.assertTrue(json_path.exists(), "Relatório JSON não foi gerado.")
        with open(json_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        expected_keys = [
            "entrada", "saida", "dispositivo", "resolucao",
            "fps_entrada", "fps_saida", "frames_processados",
            "duracao_video_s", "tempo_processamento_s",
            "pico_ram_gb", "cpu_media_pct", "deteccoes_por_classe",
        ]
        for key in expected_keys:
            self.assertIn(key, report)

        self.assertEqual(report["fps_saida"], 30.0)
        self.assertIn("Torre treliçada", report["deteccoes_por_classe"])

    @patch("processar_video.get_ffmpeg_executable", return_value=None)
    @patch("processar_video.load_model", return_value=FakeYOLO())
    def test_encoder_fallback(self, mock_load, mock_ffmpeg):
        """6. Fallback para cv2.VideoWriter quando FFmpeg não está disponível."""
        src_video = self.dir_path / "fallback_entrada.mp4"
        dst_video = self.dir_path / "fallback_saida.mp4"
        create_synthetic_video(src_video, frame_count=20, fps=20.0, width=160, height=120)

        report = process_video(
            input_path=src_video,
            output_path=dst_video,
            model_path="best.pt",
            device="cpu",
            include_audio=False,
            generate_report=False,
        )

        self.assertIsNotNone(report)
        self.assertTrue(dst_video.exists(), "Vídeo não foi gerado no fallback OpenCV.")
        cap = cv2.VideoCapture(str(dst_video))
        self.assertTrue(cap.isOpened())
        self.assertGreater(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
        cap.release()

    @patch("processar_video.load_model", return_value=FakeYOLO())
    def test_high_fps_reader_optimization(self, mock_load):
        """7. Otimização de leitura em 60 FPS: reamostra para 30 FPS inferindo metade dos frames."""
        src_video = self.dir_path / "entrada_60fps.mp4"
        dst_video = self.dir_path / "saida_60_para_30.mp4"
        # 60 frames a 60 FPS = 1.0 segundo
        create_synthetic_video(src_video, frame_count=60, fps=60.0, width=160, height=120)

        report = process_video(
            input_path=src_video,
            output_path=dst_video,
            model_path="best.pt",
            device="cpu",
            target_fps=30.0,
            include_audio=False,
            generate_report=False,
        )

        # Em 1s de vídeo a 60 FPS reamostrado para 30 FPS, ~30 frames devem ser inferidos e emitidos
        detected_count = report["deteccoes_por_classe"].get("Torre treliçada", 0)
        self.assertAlmostEqual(detected_count, 30, delta=2)

        cap = cv2.VideoCapture(str(dst_video))
        out_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.assertAlmostEqual(out_frames, 30, delta=2)

    def test_vram_oom_recovery_to_cpu(self):
        """8. Recuperação de OOM de VRAM: reduz lote pela metade até 1 e depois cai para CPU."""
        class OOMModel:
            def __init__(self):
                self.names = {0: "cable", 1: "tower_lattice"}
                self.devices_seen = []
                self.batch_sizes_seen = []

            def predict(self, frames, device="cpu", **kwargs):
                self.devices_seen.append(device)
                self.batch_sizes_seen.append(len(frames))
                if str(device).startswith("cuda"):
                    raise RuntimeError("CUDA out of memory")
                return [FakeResult([[10, 10, 50, 50, 0.88, 1]]) for _ in frames]

        oom_model = OOMModel()
        src_video = self.dir_path / "oom_in.mp4"
        dst_video = self.dir_path / "oom_out.mp4"
        create_synthetic_video(src_video, frame_count=6, fps=30.0, width=160, height=120)

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_name", return_value="Fake RTX 2060 SUPER"), \
             patch("torch.cuda.empty_cache"), \
             patch("torch.cuda.max_memory_allocated", return_value=0), \
             patch("processar_video.check_nvenc_available", return_value=False):
            report = process_video(
                input_path=src_video,
                output_path=dst_video,
                model_path="best.pt",
                device="cuda",
                batch_size=4,
                include_audio=False,
                generate_report=False,
                model_loader=lambda _: oom_model,
            )

        self.assertTrue(dst_video.exists())
        self.assertEqual(report["deteccoes_por_classe"].get("Torre treliçada"), 6)
        # Deve ter tentado lote 4 -> 2 -> 1 em CUDA e finalmente caído para CPU
        self.assertIn("cpu", oom_model.devices_seen)


if __name__ == "__main__":
    unittest.main()
