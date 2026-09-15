"""Check local YOLO detection labels before spending time on training."""
import hashlib
import math
from pathlib import Path

import yaml
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_dataset(path: Path) -> dict:
    """Resolve dataset paths independently of the shell and Ultralytics settings."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("data.yaml deve conter um mapa de configuracao.")
    names = data.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names or set(names) != set(range(len(names))):
        raise ValueError("names deve usar IDs consecutivos iniciando em 0.")
    if any(not isinstance(name, str) or not name.strip() for name in names.values()):
        raise ValueError("Todas as classes precisam de um nome.")
    if len(set(names.values())) != len(names):
        raise ValueError("Os nomes das classes devem ser diferentes.")
    root = (path.resolve().parent / data.get("path", ".")).resolve()
    result = {"path": str(root), "names": names}
    for split in ("train", "val", "test"):
        value = data.get(split)
        if value is None and split == "test":
            continue
        if not isinstance(value, str):
            raise ValueError(f"{split}: informe uma pasta images/{split}.")
        result[split] = str((root / value).resolve())
    return result


def label_path(image: Path, image_root: Path) -> Path:
    # Our scaffold deliberately uses images/<split> and labels/<split>.
    if image_root.parent.name != "images":
        raise ValueError("Use a estrutura images/<split> e labels/<split>.")
    labels = image_root.parent.parent / "labels" / image_root.name
    return (labels / image.relative_to(image_root)).with_suffix(".txt")


def check_dataset(data: dict, require_test: bool = False) -> tuple[dict, list[str]]:
    """Require explicit empty labels for reviewed negative images, preventing silent omissions."""
    errors = []
    report = {}
    seen_images = {}
    for split in ("train", "val", "test"):
        required = split != "test" or require_test
        folder = Path(data.get(split, "__missing_split__"))
        images = sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS) if folder.is_dir() else []
        counts = {name: 0 for name in data["names"].values()}
        report[split] = {"images": len(images), "objects": counts}
        if not images and required:
            errors.append(f"{split}: nenhuma imagem em {folder}.")
        labels_seen = set()
        for image in images:
            try:
                with Image.open(image) as opened:
                    opened.verify()
                digest = hashlib.sha256(image.read_bytes()).hexdigest()
                if digest in seen_images and seen_images[digest][0] != split:
                    errors.append(f"Imagem duplicada entre splits: {image} e {seen_images[digest][1]}.")
                seen_images[digest] = (split, image)
                label = label_path(image, folder)
                if label in labels_seen:
                    errors.append(f"Duas imagens compartilham a mesma anotacao: {label}.")
                labels_seen.add(label)
                if not label.is_file():
                    errors.append(f"Anotacao ausente: {label}. Para imagem negativa revisada, crie um TXT vazio.")
                    continue
                for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        fields = line.split()
                        if len(fields) != 5:
                            raise ValueError("esperado: classe centro_x centro_y largura altura")
                        class_id = int(fields[0])
                        x, y, width, height = map(float, fields[1:])
                        if class_id not in data["names"]:
                            raise ValueError("ID de classe inexistente")
                        if not all(math.isfinite(value) for value in (x, y, width, height)):
                            raise ValueError("coordenada nao finita")
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                            raise ValueError("coordenadas devem ser normalizadas entre 0 e 1, com tamanho positivo")
                        counts[data["names"][class_id]] += 1
                    except ValueError as exc:
                        errors.append(f"{label}:{line_number}: {exc}.")
            except (OSError, ValueError) as exc:
                errors.append(f"{image}: {exc}")
        if required and images:
            for name, count in counts.items():
                if not count:
                    errors.append(f"{split}: nenhuma anotacao da classe '{name}'.")
        if folder.parent.name == "images":
            labels_folder = folder.parent.parent / "labels" / folder.name
            for label in labels_folder.rglob("*.txt"):
                if label not in labels_seen:
                    errors.append(f"Anotacao sem imagem correspondente: {label}.")
    return report, errors
