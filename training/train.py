"""Windows-friendly training entry point: python -m training.train --help."""
import argparse
import logging
from pathlib import Path
import tempfile

import yaml

from config import ROOT
from detector import select_device
from training.dataset import check_dataset, read_dataset

logger = logging.getLogger(__name__)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Use um numero maior que zero.")
    return number


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Treinar YOLO para fios, postes, torres, construcoes e passaros.")
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/obstaculos/data.yaml")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="Somente conferir imagens e anotacoes; nao carrega YOLO.")
    action.add_argument("--evaluate", type=Path, metavar="BEST_PT", help="Avaliar pesos no conjunto test, sem treinar.")
    parser.add_argument("--model", type=Path, default=ROOT / "yolo11n.pt")
    parser.add_argument("--epochs", type=positive_int, default=100)
    parser.add_argument("--imgsz", type=positive_int, default=960)
    parser.add_argument("--batch", type=positive_int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda ou cuda:N")
    args = parser.parse_args(argv)
    if args.imgsz % 32:
        parser.error("--imgsz deve ser multiplo de 32.")
    if args.device not in {"auto", "cpu", "cuda"} and not (
        args.device.startswith("cuda:") and args.device[5:].isdigit()
    ):
        parser.error("--device deve ser auto, cpu, cuda ou cuda:N.")
    try:
        data = read_dataset(args.data)
        report, errors = check_dataset(data, require_test=args.evaluate is not None)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        logger.error("Nao foi possivel ler o dataset: %s", exc)
        return 1
    for split, values in report.items():
        logger.info("%s: %s imagens; objetos por classe: %s", split, values["images"], values["objects"])
    if errors:
        for message in errors[:30]:
            logger.error(message)
        if len(errors) > 30:
            logger.error("... e mais %s problemas.", len(errors) - 30)
        logger.error("Dataset ainda nao esta pronto. Veja training/README.md.")
        return 1
    if args.check:
        logger.info("Dataset validado. Nenhum treinamento iniciado.")
        return 0
    weights = (args.evaluate if args.evaluate is not None else args.model).resolve()
    if not weights.is_file() or weights.suffix.lower() != ".pt":
        logger.error("Pesos .pt locais nao encontrados: %s", weights)
        return 1

    from ultralytics import YOLO

    device = select_device(args.device)
    logger.info("Device: %s. Tamanho: %s. Batch: %s.", device, args.imgsz, args.batch)
    if device == "cpu":
        logger.info("Treinamento em CPU pode demorar; ajuste os parametros ou use uma maquina com CUDA.")
    try:
        # A persistent resolved YAML keeps checkpoints resumable and independent of cwd.
        project = ROOT / "runs" / "training"
        project.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", prefix="dataset-", dir=project,
                                         encoding="utf-8", delete=False) as resolved:
            yaml.safe_dump(data, resolved, allow_unicode=True, sort_keys=False)
            dataset_path = resolved.name
        model = YOLO(str(weights))
        common = dict(data=dataset_path, imgsz=args.imgsz, batch=args.batch, device=device,
                      workers=0, project=str(project), exist_ok=False)
        if args.evaluate:
            model.val(split="test", name="avaliacao", **common)
            logger.info("Avaliacao concluida. Consulte metricas por classe na pasta indicada pelo YOLO.")
        else:
            # No automatic AMP-check download; FP32 is a reproducible initial baseline.
            model.train(epochs=args.epochs, name="obstaculos", seed=42, amp=False, **common)
            best = Path(model.trainer.best)
            logger.info("Melhores pesos: %s", best)
            logger.info("Avalie antes de copiar para models/best.pt. O modelo ativo nao foi alterado.")
    except Exception:
        logger.exception("Falha na execucao do YOLO.")
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
