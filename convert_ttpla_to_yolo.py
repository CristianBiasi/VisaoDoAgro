"""Converte anotações TTPLA (LabelMe JSON) para formato YOLO (bounding boxes .txt).
Uso: python convert_ttpla_to_yolo.py --input <pasta_com_jpgs_e_jsons> --output <pasta_yolo>
"""
import json
import os
import shutil
import argparse
from pathlib import Path

# Mapeamento de classes TTPLA → YOLO (ignorando 'void')
CLASS_MAP = {
    "cable": 0,
    "tower_lattice": 1,
    "tower_tucohy": 2,
    "tower_wooden": 3,
}

def convert_labelme_to_yolo(json_path, img_width, img_height):
    """Converte um JSON LabelMe em linhas YOLO (classe x_center y_center w h)."""
    with open(json_path, "r") as f:
        data = json.load(f)
    
    lines = []
    for shape in data.get("shapes", []):
        label = shape.get("label", "").lower().strip()
        if label not in CLASS_MAP:
            continue  # Ignora 'void' e labels desconhecidos
        
        class_id = CLASS_MAP[label]
        points = shape["points"]
        
        # Extrair bounding box dos pontos do polígono
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Converter para formato YOLO (normalizado 0-1)
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        w = (x_max - x_min) / img_width
        h = (y_max - y_min) / img_height
        
        # Garantir que os valores estão entre 0 e 1
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        w = max(0, min(1, w))
        h = max(0, min(1, h))
        
        if w > 0.001 and h > 0.001:  # Ignorar boxes minúsculos
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Pasta com as imagens e JSONs do TTPLA")
    parser.add_argument("--output", required=True, help="Pasta de saída no formato YOLO")
    parser.add_argument("--splits", default=None, help="Pasta com train.txt, val.txt, test.txt")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Carregar splits se disponível
    splits = {"train": set(), "val": set(), "test": set()}
    splits_dir = Path(args.splits) if args.splits else None
    
    if splits_dir and splits_dir.exists():
        for split_name in splits:
            txt_file = splits_dir / f"{split_name}.txt"
            if txt_file.exists():
                with open(txt_file) as f:
                    for line in f:
                        name = line.strip().replace(".json", "")
                        splits[split_name].add(name)
        print(f"Splits carregados: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    
    # Criar estrutura YOLO
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Processar cada JSON
    json_files = sorted(input_dir.glob("*.json"))
    total = len(json_files)
    converted = 0
    skipped = 0
    
    for json_path in json_files:
        name = json_path.stem  # ex: "53_00231"
        
        # Encontrar a imagem correspondente
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = input_dir / f"{name}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        
        if not img_path:
            skipped += 1
            continue
        
        # Ler dimensões da imagem
        from PIL import Image
        with Image.open(img_path) as img:
            img_width, img_height = img.size
        
        # Converter anotações
        yolo_lines = convert_labelme_to_yolo(json_path, img_width, img_height)
        
        # Determinar o split
        if name in splits["val"]:
            split = "val"
        elif name in splits["test"]:
            split = "test"
        elif name in splits["train"]:
            split = "train"
        elif splits_dir:
            split = "train"  # Fallback para imagens sem split definido
        else:
            # Sem splits, colocar tudo em train (depois dividir manualmente)
            split = "train"
        
        # Copiar imagem
        shutil.copy2(img_path, output_dir / "images" / split / img_path.name)
        
        # Salvar labels YOLO
        label_file = output_dir / "labels" / split / f"{name}.txt"
        with open(label_file, "w") as f:
            f.write("\n".join(yolo_lines))
        
        converted += 1
    
    print(f"\nConversão concluída!")
    print(f"  Total JSONs encontrados: {total}")
    print(f"  Convertidos: {converted}")
    print(f"  Sem imagem correspondente: {skipped}")
    
    # Criar dataset.yaml
    yaml_content = f"""# TTPLA Dataset - YOLO Format
# Transmission Towers and Power Lines

path: .  # Raiz do dataset (será ajustada no Colab)
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: cable
  1: tower_lattice
  2: tower_tucohy
  3: tower_wooden

nc: 4
"""
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\nArquivo dataset.yaml criado em: {yaml_path}")


if __name__ == "__main__":
    main()
