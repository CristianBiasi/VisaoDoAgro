# VisaoDoAgro

Backend inicial para um sistema de detecção de obstáculos para drones agrícolas, organizado em arquitetura em camadas.

## Estrutura

```text
backend/
├── api/routes/
├── services/
├── domain/
├── infrastructure/
├── repositories/
├── core/
└── tests/{unit,integration}
```

## Instalação

Recomenda-se criar e ativar um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No Windows, use `.venv\\Scripts\\activate` para ativar o ambiente virtual.

## Configuração

As configurações podem ser fornecidas por variáveis de ambiente:

- `YOLO_MODEL_PATH`: caminho do modelo YOLO.
- `DEFAULT_VIDEO_SOURCE`: fonte de vídeo padrão.
- `LOG_LEVEL`: nível de log.

Exemplo:

```bash
export YOLO_MODEL_PATH=models/yolo.pt
export DEFAULT_VIDEO_SOURCE=0
export LOG_LEVEL=INFO
```

## Execução

Na raiz do projeto, execute:

```bash
uvicorn backend.main:app --reload
```

A API ficará disponível em `http://127.0.0.1:8000`. O endpoint inicial de verificação está em `GET /health`.
