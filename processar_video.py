"""Processa vídeos de torres elétricas com YOLO (best.pt) e salva o resultado anotado.

Sem interface gráfica própria — diálogos nativos do Windows para abrir e salvar.
Gera vídeo fluido a 30 FPS constantes, com controle rígido de CPU (~50%),
RAM (máx 10 GB) e aceleração por GPU (CUDA / NVENC) quando disponível.

Uso:
    python processar_video.py                                  # abre diálogos
    python processar_video.py --input video.mp4 --output saida.mp4  # sem diálogos
"""
import os
from pathlib import Path
import sys

# 1.1 Limites de recursos: aplicar ANTES de importar torch, cv2 e ultralytics
CPU_FRACTION_DEFAULT = 0.5
LOGICAL_CPUS = os.cpu_count() or 2
CPU_THREADS_DEFAULT = max(1, int(LOGICAL_CPUS * CPU_FRACTION_DEFAULT))
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, str(CPU_THREADS_DEFAULT))

# Ultralytics settings: evitar erro de permissão no sandbox do Windows
if "YOLO_CONFIG_DIR" not in os.environ:
    _local_cfg_parent = Path(__file__).resolve().parent / ".ultralytics"
    _local_cfg_sub = _local_cfg_parent / "Ultralytics"
    try:
        _local_cfg_sub.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(_local_cfg_parent)
    except Exception:
        pass

import argparse
import hashlib
import json
import logging
import queue
import shutil
import subprocess
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np
import psutil
import torch

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get("total", 0)
            self.n = 0
        def update(self, n=1):
            self.n += n
        def set_postfix(self, *args, **kwargs):
            pass
        def set_postfix_str(self, *args, **kwargs):
            pass
        def close(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

logger = logging.getLogger("processar_video")

CLASS_TRANSLATIONS = {
    "cable": "Cabo",
    "tower_lattice": "Torre treliçada",
    "tower_tucohy": "Torre tubular/concreto",
    "tower_wooden": "Torre de madeira",
}

CLASS_COLORS = {
    "cable": (0, 215, 255),         # Amarelo (BGR)
    "tower_lattice": (60, 60, 255),  # Vermelho (BGR)
    "tower_tucohy": (255, 140, 0),   # Azul (BGR)
    "tower_wooden": (40, 160, 40),   # Verde (BGR)
}


def get_class_color(cls_name: str) -> tuple[int, int, int]:
    """Retorna cor BGR definida para a classe ou uma cor estável via hash."""
    if cls_name in CLASS_COLORS:
        return CLASS_COLORS[cls_name]
    h = int(hashlib.md5(cls_name.encode("utf-8")).hexdigest()[:6], 16)
    return ((h & 0xFF), ((h >> 8) & 0xFF), ((h >> 16) & 0xFF))


def get_class_label(cls_name: str) -> str:
    """Retorna o rótulo em português correspondente ou o nome original."""
    return CLASS_TRANSLATIONS.get(cls_name, cls_name)


def apply_resource_limits(cpu_fraction: float = 0.5) -> int:
    """Aplica limites de threads e afinidade de CPU para respeitar a fração da CPU."""
    threads = max(1, int(LOGICAL_CPUS * cpu_fraction))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cv2.setNumThreads(max(1, threads // 2))

    proc = psutil.Process()
    try:
        proc.cpu_affinity(list(range(threads)))
    except Exception as exc:
        logger.debug("Não foi possível definir cpu_affinity: %s", exc)

    try:
        proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception as exc:
        logger.debug("Não foi possível definir prioridade: %s", exc)

    return threads


def calculate_queue_size(width: int, height: int, max_ram_gb: float = 10.0) -> int:
    """Calcula tamanho seguro da fila de frames com base na resolução e limite de RAM."""
    bytes_per_frame = max(1, width * height * 3)
    target_bytes = max_ram_gb * 0.25 * (1024 ** 3)
    calculated = int(target_bytes / bytes_per_frame / 2)
    return min(64, max(8, calculated))


def frames_to_emit(src_ts: float, next_out_ts: float, src_fps: float, out_fps: float = 30.0) -> tuple[int, float]:
    """Calcula quantos frames de saída devem ser emitidos para manter duração exata e out_fps constante."""
    tolerance = 1.0 / max(1.0, src_fps) / 2.0
    out_interval = 1.0 / max(1.0, out_fps)
    count = 0
    curr = next_out_ts
    while curr <= src_ts + tolerance:
        count += 1
        curr += out_interval
    return count, curr


def draw_detections(
    frame: np.ndarray,
    detections: list[Any] | np.ndarray,
    class_names: dict[int, str] | None = None,
    hud_info: dict[str, Any] | None = None,
) -> np.ndarray:
    """Desenha caixas delimitadoras, rótulos e HUD sobre o frame."""
    output = frame.copy()
    h, w = output.shape[:2]
    class_names = class_names or {}

    thickness = max(2, w // 640)
    font_scale = max(0.45, w / 1600.0)

    for det in detections:
        if len(det) < 6:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in det[:4]]
        conf = float(det[4])
        cls_id = int(det[5])
        cls_raw_name = class_names.get(cls_id, str(cls_id))
        cls_pt_name = get_class_label(cls_raw_name)
        color = get_class_color(cls_raw_name)

        # Retângulo da detecção
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        # Rótulo com fundo preenchido
        label = f"{cls_pt_name} {int(round(conf * 100))}%"
        if len(det) >= 7 and det[6] is not None and int(det[6]) >= 0:
            label = f"#{int(det[6])} " + label

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_y1 = max(0, y1 - th - baseline - 4)
        label_y2 = y1
        label_x2 = min(w, x1 + tw + 6)
        cv2.rectangle(output, (x1, label_y1), (label_x2, label_y2), color, -1)

        text_color = (0, 0, 0) if (color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299) > 130 else (255, 255, 255)
        cv2.putText(
            output,
            label,
            (x1 + 3, label_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            max(1, thickness // 2),
            cv2.LINE_AA,
        )

    if hud_info:
        _draw_hud(output, hud_info)

    return output


def _draw_hud(frame: np.ndarray, hud_info: dict[str, Any]):
    """Renderiza painel HUD semitransparente no canto superior esquerdo."""
    h, w = frame.shape[:2]
    scale = max(0.4, w / 1800.0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    frame_idx = hud_info.get("frame_idx", 0)
    total_frames = hud_info.get("total_frames", 0)
    ts = hud_info.get("timestamp_s", 0.0)
    m = int(ts // 60)
    s = int(ts % 60)
    time_str = f"{m:02d}:{s:02d}"

    line1 = f"Frame {frame_idx}/{total_frames} · {time_str}"
    counts = hud_info.get("counts", {})
    if counts:
        counts_str = " · ".join(f"{k}: {v}" for k, v in counts.items())
    else:
        counts_str = "Nenhuma detecção"
    model_name = hud_info.get("model_name", "best.pt")
    line3 = f"Modelo: {model_name}"

    lines = [line1, counts_str, line3]
    pad = 8
    max_tw = 0
    total_th = 0
    line_heights = []
    for l in lines:
        (tw, th), base = cv2.getTextSize(l, font, scale, 1)
        if tw > max_tw:
            max_tw = tw
        lh = th + base + 4
        line_heights.append(lh)
        total_th += lh

    hud_w = max_tw + pad * 2
    hud_h = total_th + pad * 2
    hud_x = 14
    hud_y = 14

    roi = frame[hud_y : hud_y + hud_h, hud_x : hud_x + hud_w]
    if roi.shape[0] == hud_h and roi.shape[1] == hud_w:
        sub = np.zeros_like(roi, dtype=np.uint8)
        cv2.addWeighted(roi, 0.25, sub, 0.75, 0, roi)
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (60, 75, 80), 1)

    y_offset = hud_y + pad
    for i, l in enumerate(lines):
        y_offset += line_heights[i] - 4
        cv2.putText(
            frame,
            l,
            (hud_x + pad, y_offset),
            font,
            scale,
            (230, 245, 240),
            1,
            cv2.LINE_AA,
        )


def select_device(requested: str = "auto") -> tuple[str, bool]:
    """Retorna (device, half). 'auto' prefere CUDA se disponível."""
    req = requested.lower().strip()
    if req in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda:0", True
    if req == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA indisponível — driver NVIDIA ou PyTorch CUDA não detectado. Usando CPU.")
    return "cpu", False


def check_nvenc_available(ffmpeg_exe: str) -> bool:
    """Verifica se o encoder h264_nvenc funciona no hardware atual."""
    try:
        cmd = [
            ffmpeg_exe, "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
            "-c:v", "h264_nvenc",
            "-f", "null", "-",
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return res.returncode == 0
    except Exception:
        return False


def get_ffmpeg_executable() -> str | None:
    """Obtém o binário ffmpeg via imageio_ffmpeg ou shutil.which."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return shutil.which("ffmpeg")


def ask_input_video() -> Path | None:
    """Abre diálogo nativo para selecionar vídeo de entrada."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Selecione o vídeo das torres",
            filetypes=[
                ("Vídeos", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"),
                ("Todos os arquivos", "*.*"),
            ],
        )
        root.destroy()
        return Path(path) if path else None
    except Exception as exc:
        print(f"Interface gráfica indisponível ({exc}).")
        user_input = input("Digite o caminho do vídeo de entrada: ").strip().strip('"')
        return Path(user_input) if user_input else None


def ask_output_path(default_path: Path) -> Path | None:
    """Abre diálogo nativo para escolher onde salvar o vídeo processado."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.asksaveasfilename(
            title="Salvar vídeo processado como",
            defaultextension=".mp4",
            initialfile=default_path.name,
            initialdir=str(default_path.parent),
            filetypes=[("Vídeo MP4", "*.mp4"), ("Todos os arquivos", "*.*")],
        )
        root.destroy()
        return Path(path) if path else None
    except Exception as exc:
        print(f"Interface gráfica indisponível ({exc}).")
        user_input = input(f"Salvar em [{default_path}]: ").strip().strip('"')
        return Path(user_input) if user_input else default_path


def load_model(model_path: str):
    """Carrega o modelo YOLO (ponto de injeção para testes)."""
    from ultralytics import YOLO
    return YOLO(model_path)


class ResourceMonitor(threading.Thread):
    """Monitor de consumo de CPU, RAM e VRAM."""

    def __init__(self, max_ram_gb: float, pause_event: threading.Event, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.max_ram_gb = max_ram_gb
        self.pause_event = pause_event
        self.stop_event = stop_event
        self.peak_ram_bytes = 0
        self.peak_vram_bytes = 0
        self.cpu_samples: list[float] = []
        self.oom_triggered = False
        self.reduce_batch_to_one = False

    def run(self):
        proc = psutil.Process()
        limit_bytes = self.max_ram_gb * (1024 ** 3)
        pause_bytes = limit_bytes * 0.85
        resume_bytes = limit_bytes * 0.70

        # Inicializa o baseline do cpu_percent
        try:
            proc.cpu_percent(interval=None)
        except Exception:
            pass

        while not self.stop_event.is_set():
            try:
                total_rss = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    try:
                        total_rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                if total_rss > self.peak_ram_bytes:
                    self.peak_ram_bytes = total_rss

                cpu_pct = proc.cpu_percent(interval=None) / max(1, LOGICAL_CPUS)
                if cpu_pct > 0.0 or not self.cpu_samples:
                    self.cpu_samples.append(cpu_pct)

                if torch.cuda.is_available():
                    try:
                        vram = torch.cuda.max_memory_allocated()
                        if vram > self.peak_vram_bytes:
                            self.peak_vram_bytes = vram
                    except Exception:
                        pass

                if total_rss > limit_bytes:
                    self.oom_triggered = True
                    self.stop_event.set()
                    break
                elif total_rss > pause_bytes:
                    self.reduce_batch_to_one = True
                    self.pause_event.clear()
                elif total_rss < resume_bytes:
                    self.pause_event.set()
            except Exception:
                pass
            self.stop_event.wait(timeout=1.0)


def process_video(
    input_path: Path,
    output_path: Path,
    model_path: str = "best.pt",
    device: str = "auto",
    conf: float = 0.35,
    iou: float = 0.5,
    imgsz: int = 640,
    batch_size: int | None = None,
    target_fps: float = 30.0,
    cpu_fraction: float = 0.5,
    max_ram_gb: float = 10.0,
    tracking: bool = False,
    include_audio: bool = True,
    draw_hud: bool = True,
    generate_report: bool = True,
    model_loader: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Pipeline completo de leitura, inferência e escrita com limites de recursos."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Vídeo de entrada não encontrado: {input_path}")

    cpu_threads = apply_resource_limits(cpu_fraction)
    selected_dev, half = select_device(device)

    # Lote padrão (em modo tracking, inferência é frame a frame com lote 1)
    if tracking:
        batch_size = 1
    elif batch_size is None:
        batch_size = 8 if selected_dev.startswith("cuda") else 1

    # Resolução e metadados
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {input_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    duration_s = total_frames / max(1.0, src_fps)
    queue_frames = calculate_queue_size(src_w, src_h, max_ram_gb)

    # Imprimir informações
    if selected_dev.startswith("cuda"):
        dev_desc = f"{torch.cuda.get_device_name(0)} (CUDA, FP16)"
    else:
        dev_desc = f"CPU ({cpu_threads} threads)"

    print("=" * 64)
    print("AGROSAFE VISION — PROCESSAMENTO DE TORRES ELÉTRICAS")
    print(f"Vídeo de entrada : {input_path.name} ({src_w}x{src_h}, {src_fps:.2f} FPS, {total_frames} frames)")
    print(f"Dispositivo      : {dev_desc}")
    print(f"Modelo           : {Path(model_path).name}")
    print(f"Lote             : {batch_size} | Resolução YOLO: {imgsz}")
    print(f"Limite CPU       : {int(cpu_fraction * 100)}% ({cpu_threads} threads) | Limite RAM: {max_ram_gb:.1f} GB")
    print("=" * 64)

    # Carregar modelo
    actual_loader = model_loader if model_loader is not None else load_model
    model = actual_loader(model_path)
    class_names = getattr(model, "names", {})
    if isinstance(class_names, (list, tuple)):
        class_names = dict(enumerate(class_names))

    # Warmup
    try:
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        model.predict([dummy], imgsz=imgsz, device=selected_dev, half=half, verbose=False)
    except Exception as exc:
        logger.debug("Aquecimento ignorado: %s", exc)

    # Preparar arquivo temporário
    temp_dir = input_path.parent
    temp_file = temp_dir / f".{input_path.stem}_processando.mp4"
    if temp_file.exists():
        try:
            temp_file.unlink()
        except OSError:
            pass

    # Configurar Encoder
    ffmpeg_exe = get_ffmpeg_executable()
    use_nvenc = False
    writer_proc = None
    cv2_writer = None

    # Garantir dimensões pares para YUV420P
    out_w = src_w + (src_w % 2)
    out_h = src_h + (src_h % 2)

    if ffmpeg_exe:
        if selected_dev.startswith("cuda"):
            use_nvenc = check_nvenc_available(ffmpeg_exe)
        encoder_name = "h264_nvenc (GPU)" if use_nvenc else "libx264 (CPU)"
        print(f"Encoder          : {encoder_name} via FFmpeg")

        cmd = [
            ffmpeg_exe, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{out_w}x{out_h}",
            "-r", str(target_fps),
            "-i", "-",
        ]
        if include_audio:
            cmd.extend(["-i", str(input_path)])

        if use_nvenc:
            cmd.extend(["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "23"])
        else:
            cmd.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-threads", str(cpu_threads)])

        cmd.extend(["-pix_fmt", "yuv420p"])

        if include_audio:
            cmd.extend(["-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-shortest"])

        cmd.extend(["-movflags", "+faststart", str(temp_file)])
        try:
            writer_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as exc:
            logger.warning("Falha ao iniciar FFmpeg (%s). Usando cv2.VideoWriter como fallback.", exc)
            writer_proc = None

    if writer_proc is None:
        print("Encoder          : cv2.VideoWriter mp4v (fallback sem áudio)")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cv2_writer = cv2.VideoWriter(str(temp_file), fourcc, target_fps, (src_w, src_h))
        if not cv2_writer.isOpened():
            raise RuntimeError(f"Não foi possível abrir cv2.VideoWriter para {temp_file}")

    # Filas e eventos de sincronização
    frame_queue: queue.Queue = queue.Queue(maxsize=queue_frames)
    writer_queue: queue.Queue = queue.Queue(maxsize=queue_frames)

    pause_event = threading.Event()
    pause_event.set()
    stop_event = threading.Event()
    error_holder: list[Exception] = []

    # Monitor de recursos
    monitor = ResourceMonitor(max_ram_gb, pause_event, stop_event)
    monitor.start()

    # THREAD LEITORA
    def reader_worker():
        try:
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise RuntimeError(f"Falha ao reabrir {input_path}")

            idx = 0
            reader_next_out_ts = 0.0
            while not stop_event.is_set():
                pause_event.wait()
                ret, frame = cap.read()
                if not ret:
                    break

                ts_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                ts_s = (ts_msec / 1000.0) if (ts_msec and ts_msec > 0) else (idx / max(1.0, src_fps))

                # Se o vídeo tiver FPS superior ao de saída (ex: 60 -> 30), otimiza pulando a inferência
                # dos frames que não serão emitidos no vídeo final
                k_emit, reader_next_out_ts = frames_to_emit(ts_s, reader_next_out_ts, src_fps, target_fps)
                skip = (src_fps > target_fps) and (k_emit == 0)

                frame_queue.put((idx, ts_s, frame, skip))
                idx += 1

            cap.release()
        except Exception as exc:
            error_holder.append(exc)
            stop_event.set()
        finally:
            frame_queue.put(None)

    reader_thread = threading.Thread(target=reader_worker, daemon=True)
    reader_thread.start()

    # THREAD ESCRITORA
    written_frames_count = [0]

    def writer_worker():
        try:
            next_out_ts = 0.0
            while True:
                if stop_event.is_set() and writer_queue.empty():
                    break
                try:
                    item = writer_queue.get(timeout=0.2)
                except queue.Empty:
                    if stop_event.is_set():
                        break
                    continue

                if item is None:
                    break

                idx, ts_s, frame_bgr, dets, counts = item

                # Calcular emissões para 30 FPS constantes ANTES de desenhar overlay
                k, next_out_ts = frames_to_emit(ts_s, next_out_ts, src_fps, target_fps)
                if k > 0:
                    hud_data = None
                    if draw_hud:
                        hud_data = {
                            "frame_idx": idx + 1,
                            "total_frames": total_frames,
                            "timestamp_s": ts_s,
                            "counts": counts,
                            "model_name": Path(model_path).name,
                        }

                    rendered = draw_detections(frame_bgr, dets, class_names, hud_data)

                    if writer_proc:
                        # Pad para par se necessário
                        if out_w != src_w or out_h != src_h:
                            pad_b = out_h - src_h
                            pad_r = out_w - src_w
                            rendered_to_write = cv2.copyMakeBorder(
                                rendered, 0, pad_b, 0, pad_r, cv2.BORDER_CONSTANT, value=0
                            )
                        else:
                            rendered_to_write = rendered
                        raw_bytes = rendered_to_write.tobytes()
                        for _ in range(k):
                            writer_proc.stdin.write(raw_bytes)
                            written_frames_count[0] += 1
                    elif cv2_writer:
                        for _ in range(k):
                            cv2_writer.write(rendered)
                            written_frames_count[0] += 1
        except Exception as exc:
            error_holder.append(exc)
            stop_event.set()

    writer_thread = threading.Thread(target=writer_worker, daemon=True)
    writer_thread.start()

    # THREAD PRINCIPAL: INFERÊNCIA
    detections_by_class: dict[str, int] = {}
    frames_with_detection: dict[str, int] = {}
    conf_sum_by_class: dict[str, float] = {}
    unique_ids_by_class: dict[str, set[int]] = {}

    start_time = time.perf_counter()
    pbar = tqdm(total=total_frames, desc="Processando", unit="frames")

    def _run_batch_with_oom_recovery(items_to_infer: list[tuple[int, float, np.ndarray]]):
        nonlocal curr_batch, selected_dev, half, dev_desc
        pending = list(items_to_infer)
        oom_errors = (torch.cuda.OutOfMemoryError, RuntimeError) if hasattr(torch.cuda, "OutOfMemoryError") else (RuntimeError,)

        while pending:
            chunk = pending[:curr_batch]
            try:
                _infer_and_enqueue(
                    chunk, model, selected_dev, half, imgsz, conf, iou,
                    tracking, class_names, writer_queue,
                    detections_by_class, frames_with_detection, conf_sum_by_class,
                    unique_ids_by_class,
                )
                pbar.update(len(chunk))
                pending = pending[len(chunk):]
            except oom_errors as oom_exc:
                is_oom = isinstance(oom_exc, getattr(torch.cuda, "OutOfMemoryError", ())) or "out of memory" in str(oom_exc).lower()
                if not is_oom:
                    raise
                if selected_dev.startswith("cuda") and curr_batch > 1:
                    new_batch = max(1, curr_batch // 2)
                    logger.warning("VRAM OOM detectado. Reduzindo lote de %d para %d.", curr_batch, new_batch)
                    curr_batch = new_batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                elif selected_dev.startswith("cuda") and curr_batch == 1:
                    logger.warning("VRAM OOM no lote 1. Alternando inferência para CPU.")
                    print("\n⚠ VRAM insuficiente mesmo com lote 1 — alternando para CPU.")
                    selected_dev = "cpu"
                    half = False
                    dev_desc = f"CPU ({cpu_threads} threads, fallback de VRAM)"
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise

    try:
        buffer = []
        curr_batch = batch_size

        while not stop_event.is_set():
            if monitor.reduce_batch_to_one and curr_batch > 1:
                curr_batch = 1

            item = frame_queue.get()
            if item is None:
                # Processar restante do buffer
                if buffer:
                    _run_batch_with_oom_recovery(buffer)
                    buffer.clear()
                break

            idx, ts_s, frame_bgr, skip = item
            if skip:
                writer_queue.put((idx, ts_s, frame_bgr, [], {}))
                pbar.update(1)
                continue

            buffer.append((idx, ts_s, frame_bgr))
            if len(buffer) >= curr_batch:
                _run_batch_with_oom_recovery(buffer)
                buffer.clear()

            # Atualizar status da barra com métricas
            ram_gb = monitor.peak_ram_bytes / (1024 ** 3)
            cpu_pct = monitor.cpu_samples[-1] if monitor.cpu_samples else 0.0
            if selected_dev.startswith("cuda"):
                vram_gb = monitor.peak_vram_bytes / (1024 ** 3)
                pbar.set_postfix_str(f"GPU {vram_gb:.1f}GB · RAM {ram_gb:.1f}GB · CPU {cpu_pct:.0f}%")
            else:
                pbar.set_postfix_str(f"RAM {ram_gb:.1f}GB · CPU {cpu_pct:.0f}%")

        # Conclusão normal da inferência: drena a fila de escrita
        writer_queue.put(None)
        writer_thread.join()
        reader_thread.join()

    except (KeyboardInterrupt, Exception) as exc:
        stop_event.set()
        error_holder.append(exc)
    finally:
        pbar.close()
        stop_event.set()
        monitor.stop_event.set()
        if writer_thread.is_alive():
            try:
                writer_queue.put_nowait(None)
            except Exception:
                pass
            writer_thread.join(timeout=3.0)
        if reader_thread.is_alive():
            reader_thread.join(timeout=2.0)

        # Encerrar encoder e verificar retorno do FFmpeg
        ffmpeg_stderr = ""
        ffmpeg_returncode = 0
        if writer_proc:
            if writer_proc.stdin:
                try:
                    writer_proc.stdin.close()
                except Exception:
                    pass
            if writer_proc.stderr:
                try:
                    ffmpeg_stderr = writer_proc.stderr.read().decode("utf-8", errors="replace").strip()
                    writer_proc.stderr.close()
                except Exception:
                    pass
            ffmpeg_returncode = writer_proc.wait()
            if ffmpeg_returncode != 0 and not error_holder and not monitor.oom_triggered:
                error_holder.append(
                    RuntimeError(f"FFmpeg encerrou com código {ffmpeg_returncode}: {ffmpeg_stderr}")
                )
        elif cv2_writer:
            cv2_writer.release()

    total_proc_time = time.perf_counter() - start_time

    # Tratamento de erros e cancelamento
    if monitor.oom_triggered:
        if temp_file.exists():
            temp_file.unlink()
        print("\n❌ ERRO FATAL: Limite de memória RAM ultrapassado (> 10 GB). Processamento abortado.")
        sys.exit(3)

    if error_holder:
        first_err = error_holder[0]
        if isinstance(first_err, KeyboardInterrupt):
            if temp_file.exists():
                temp_file.unlink()
            print("\n⚠ Processamento cancelado pelo usuário.")
            return {}
        else:
            if temp_file.exists():
                temp_file.unlink()
            raise first_err

    # Mover arquivo temporário para o destino final
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        try:
            output_path.unlink()
        except OSError:
            pass
    shutil.move(str(temp_file), str(output_path))

    # Compilar relatório de métricas
    avg_cpu = float(np.mean(monitor.cpu_samples)) if monitor.cpu_samples else 0.0
    avg_speed = (total_frames / max(0.001, total_proc_time))
    avg_conf_by_class = {
        cls_name: round(conf_sum_by_class[cls_name] / max(1, detections_by_class[cls_name]), 2)
        for cls_name in detections_by_class
    }

    report: dict[str, Any] = {
        "entrada": str(input_path),
        "saida": str(output_path),
        "modelo": str(model_path),
        "dispositivo": dev_desc,
        "resolucao": f"{src_w}x{src_h}",
        "fps_entrada": round(src_fps, 2),
        "fps_saida": round(target_fps, 2),
        "frames_processados": total_frames,
        "duracao_video_s": round(duration_s, 2),
        "tempo_processamento_s": round(total_proc_time, 2),
        "velocidade_media_fps": round(avg_speed, 1),
        "pico_ram_gb": round(monitor.peak_ram_bytes / (1024 ** 3), 2),
        "pico_vram_gb": round(monitor.peak_vram_bytes / (1024 ** 3), 2) if selected_dev.startswith("cuda") else 0.0,
        "cpu_media_pct": round(avg_cpu, 1),
        "deteccoes_por_classe": detections_by_class,
        "frames_com_deteccao_por_classe": frames_with_detection,
        "confianca_media_por_classe": avg_conf_by_class,
    }
    if tracking and unique_ids_by_class:
        report["ids_unicos_por_classe"] = {k: len(v) for k, v in unique_ids_by_class.items()}

    if generate_report:
        report_path = output_path.with_suffix(".json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def _infer_and_enqueue(
    batch_items: list[tuple[int, float, np.ndarray]],
    model: Any,
    device: str,
    half: bool,
    imgsz: int,
    conf: float,
    iou: float,
    tracking: bool,
    class_names: dict[int, str],
    writer_queue: queue.Queue,
    detections_by_class: dict[str, int],
    frames_with_detection: dict[str, int],
    conf_sum_by_class: dict[str, float],
    unique_ids_by_class: dict[str, set[int]] | None = None,
):
    """Executa predição em lote e encaminha resultados para a fila do escritor."""
    frames_bgr = [item[2] for item in batch_items]

    if tracking:
        # Tracking precisa rodar frame a frame com persist=True
        results = [
            model.track(f, imgsz=imgsz, conf=conf, iou=iou, device=device, half=half, persist=True, verbose=False)[0]
            for f in frames_bgr
        ]
    else:
        results = model.predict(frames_bgr, imgsz=imgsz, conf=conf, iou=iou, device=device, half=half, verbose=False)

    for i, res in enumerate(results):
        idx, ts_s, frame_bgr = batch_items[i]
        boxes = res.boxes
        dets = []
        counts: dict[str, int] = {}
        seen_in_frame: set[str] = set()

        if boxes is not None and len(boxes) > 0:
            data = boxes.data.cpu().numpy()
            for row in data:
                # row: x1, y1, x2, y2, [track_id], conf, cls
                tid = -1
                if len(row) == 7:
                    x1, y1, x2, y2, tid, c, cl = row
                    dets.append([x1, y1, x2, y2, c, cl, tid])
                else:
                    x1, y1, x2, y2, c, cl = row[:6]
                    dets.append([x1, y1, x2, y2, c, cl, -1])

                c_name = get_class_label(class_names.get(int(cl), str(int(cl))))
                counts[c_name] = counts.get(c_name, 0) + 1
                detections_by_class[c_name] = detections_by_class.get(c_name, 0) + 1
                conf_sum_by_class[c_name] = conf_sum_by_class.get(c_name, 0.0) + float(c)
                seen_in_frame.add(c_name)
                if unique_ids_by_class is not None and int(tid) >= 0:
                    unique_ids_by_class.setdefault(c_name, set()).add(int(tid))

        for c_name in seen_in_frame:
            frames_with_detection[c_name] = frames_with_detection.get(c_name, 0) + 1

        writer_queue.put((idx, ts_s, frame_bgr, dets, counts))


def main(cli_args: list[str] | None = None, model_loader: Callable[[str], Any] | None = None) -> int:
    """Ponto de entrada do script."""
    parser = argparse.ArgumentParser(description="Processador offline de vídeos de torres elétricas com YOLO")
    parser.add_argument("--input", type=str, default=None, help="Caminho do vídeo de entrada")
    parser.add_argument("--output", type=str, default=None, help="Caminho do vídeo de saída")
    parser.add_argument("--model", type=str, default="best.pt", help="Caminho do modelo YOLO (padrão: best.pt)")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda ou cpu")
    parser.add_argument("--conf", type=float, default=0.35, help="Limiar de confiança (padrão: 0.35)")
    parser.add_argument("--iou", type=float, default=0.5, help="Limiar de IOU para NMS (padrão: 0.5)")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamanho da imagem para inferência (padrão: 640)")
    parser.add_argument("--batch", type=int, default=None, help="Tamanho do lote (padrão: 8 CUDA / 1 CPU)")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS do vídeo final (padrão: 30)")
    parser.add_argument("--cpu-fraction", type=float, default=0.5, help="Fração máxima da CPU permitida (padrão: 0.5)")
    parser.add_argument("--max-ram-gb", type=float, default=10.0, help="Limite máximo de memória RAM em GB (padrão: 10)")
    parser.add_argument("--track", action="store_true", help="Ativa rastreamento com IDs estáveis")
    parser.add_argument("--no-audio", action="store_true", help="Não copia áudio do vídeo original")
    parser.add_argument("--no-hud", action="store_true", help="Não desenha painel de HUD no vídeo")
    parser.add_argument("--no-report", action="store_true", help="Não gera relatório .json")

    args = parser.parse_args(cli_args)

    # 1. Obter vídeo de entrada
    if args.input:
        in_path = Path(args.input)
    else:
        in_path = ask_input_video()
        if not in_path:
            print("Nenhum vídeo selecionado. Encerrando.")
            return 1

    if not in_path.is_file():
        print(f"Erro: Arquivo de entrada não existe: {in_path}")
        return 1

    # 2. Localizar modelo
    model_path = Path(args.model)
    if not model_path.is_file():
        alt_model = Path("models") / args.model
        if alt_model.is_file():
            model_path = alt_model
        else:
            print(f"Erro: Modelo não encontrado em '{model_path}' nem em '{alt_model}'.")
            return 1

    # 3. Definir caminho de saída
    target_out_path = None
    if args.output:
        target_out_path = Path(args.output)
    default_out_path = in_path.parent / f"{in_path.stem}_processado.mp4"

    # 4. Executar processamento (grava temporário na pasta de entrada)
    temp_target = target_out_path or default_out_path
    try:
        report = process_video(
            input_path=in_path,
            output_path=temp_target,
            model_path=str(model_path),
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            batch_size=args.batch,
            target_fps=args.fps,
            cpu_fraction=args.cpu_fraction,
            max_ram_gb=args.max_ram_gb,
            tracking=args.track,
            include_audio=not args.no_audio,
            draw_hud=not args.no_hud,
            generate_report=not args.no_report,
            model_loader=model_loader,
        )
    except Exception as exc:
        print(f"\n❌ Erro durante o processamento: {exc}")
        return 2

    if not report:
        return 0

    # 5. Salvar como (se não fornecido na CLI)
    final_path = temp_target
    if not args.output:
        chosen = ask_output_path(default_out_path)
        if not chosen:
            try:
                resp = input(f"Salvar em {default_out_path}? [S/n]: ").strip().lower()
                if resp in ("n", "nao", "não", "no"):
                    print(f"Trabalho preservado em '{default_out_path}' (nunca descartado em silêncio).")
            except Exception:
                pass
            chosen = default_out_path

        if chosen != temp_target:
            try:
                chosen.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(temp_target), str(chosen))
                old_rep = temp_target.with_suffix(".json")
                if old_rep.exists():
                    old_rep.unlink()
                final_path = chosen
                report["saida"] = str(final_path)
                if not args.no_report:
                    with open(final_path.with_suffix(".json"), "w", encoding="utf-8") as f:
                        json.dump(report, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Erro ao mover para {chosen}: {e}. Mantido em {temp_target}.")

    # 6. Resumo e opção de abrir
    print("\n" + "=" * 64)
    print("✅ PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print(f"Arquivo gerado : {final_path}")
    if not args.no_report:
        print(f"Relatório JSON : {final_path.with_suffix('.json')}")
    print(f"Tempo total    : {report.get('tempo_processamento_s', 0):.1f} s ({report.get('velocidade_media_fps', 0):.1f} FPS)")
    print(f"Pico RAM / CPU : {report.get('pico_ram_gb', 0)} GB / {report.get('cpu_media_pct', 0)}%")
    print("Detecções por classe:")
    for k, v in report.get("deteccoes_por_classe", {}).items():
        frames_c = report.get("frames_com_deteccao_por_classe", {}).get(k, 0)
        conf_m = report.get("confianca_media_por_classe", {}).get(k, 0.0)
        print(f"  • {k}: {v} objetos em {frames_c} frames (confiança média: {int(conf_m * 100)}%)")
    print("=" * 64)

    # Pergunta interativa se executado sem --output (modo usuário interativo)
    if not args.output and hasattr(os, "startfile"):
        try:
            resp = input("\nDeseja abrir o vídeo agora? [s/N]: ").strip().lower()
            if resp in ("s", "sim", "y", "yes"):
                os.startfile(str(final_path))
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
