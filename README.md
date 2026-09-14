# VisaoDoAgro

Backend para detecção de obstáculos em drones agrícolas, com pipeline de visão computacional preparado para YOLO, análise de risco, alertas em tempo real, persistência e métricas.

## Arquitetura

```text
backend/
├── api/              # FastAPI, rotas HTTP, WebSocket e dependências
├── services/         # Detecção, risco, alertas, streaming e métricas
├── domain/           # Entidades Pydantic e enums do domínio
├── infrastructure/  # YOLO, OpenCV, fonte de vídeo e notificações
├── repositories/     # Persistência SQLite
├── core/             # Configurações por ambiente
└── tests/            # Testes unitários e de integração
```

O fluxo principal é:

```text
Vídeo → VideoSource → DetectionService → RiskAnalysisService
			 → AlertService → Notifier (WebSocket + log JSON)
			 → DetectionRepository (SQLite) → MetricsService
```

O `YoloModel` é singleton via `FastAPI Depends`. Os serviços são criados por requisição e recebem suas dependências por injeção, permitindo testes com mocks e fakes.

## Requisitos e instalação

Python 3.12 ou superior e `uv` são recomendados:

```bash
uv sync
```

Alternativamente:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No Windows, ative com `.venv\\Scripts\\activate`.

## Configuração

As configurações podem ser fornecidas por variáveis de ambiente ou em um arquivo `.env`:

| Variável | Padrão | Finalidade |
| --- | --- | --- |
| `YOLO_MODEL_PATH` | `models/yolo.pt` | Caminho dos pesos YOLO |
| `DEFAULT_VIDEO_SOURCE` | `0` | Fonte de vídeo padrão |
| `LOG_LEVEL` | `INFO` | Nível de log |
| `DATABASE_PATH` | `data/detections.db` | Banco SQLite |
| `GROUND_TRUTH_PATH` | `tests/fixtures/ground_truth.json` | Anotações para métricas |

Para usar um modelo YOLO real:

```bash
export YOLO_MODEL_PATH=models/yolo11n.pt
```

Os testes não carregam pesos reais: usam fakes e vídeos temporários.

## Execução da API

```bash
uv run uvicorn backend.main:app --reload
```

Ou, com o ambiente ativado:

```bash
uvicorn backend.main:app --reload
```

A API fica em `http://127.0.0.1:8000`. A documentação Swagger está em `/docs`.

### Endpoints principais

- `GET /health`: verifica a disponibilidade da API.
- `POST /detect/image`: recebe upload de imagem e retorna detecções e alertas.
- `POST /detect/video`: recebe upload de vídeo e retorna um resumo.
- `POST /detect/stream/start`: processa um vídeo local por caminho.
- `GET /metrics/report`: calcula métricas usando o banco e o ground truth configurado.
- `WebSocket /ws/alerts`: recebe alertas durante o processamento contínuo.

Exemplo de processamento contínuo:

```bash
curl -X POST http://127.0.0.1:8000/detect/stream/start \
	-H 'Content-Type: application/json' \
	-d '{"video_path":"tests/fixtures/sample.avi"}'
```

Resposta esperada:

```json
{
	"total_frames": 3,
	"average_fps": 120.5,
	"total_alerts": 3
}
```

O arquivo de anotações pode ser JSON ou CSV. O formato JSON esperado é:

```json
[
	{
		"frame_id": "frame-1",
		"obstacle_class": "POSTE",
		"bounding_box": {
			"x_min": 5,
			"y_min": 5,
			"x_max": 30,
			"y_max": 30
		}
	}
]
```

Para consultar outro conjunto de anotações:

```bash
curl 'http://127.0.0.1:8000/metrics/report?annotation_path=tests/fixtures/ground_truth.json'
```

## Testes

```bash
uv run pytest tests/unit tests/integration
```

O teste end-to-end executa vídeo, detecção fake, análise de risco, envio WebSocket, log, persistência SQLite e cálculo de métricas em uma única sequência.

## Docker

```bash
docker build -t visaodoagro .
docker run --rm -p 8000:8000 visaodoagro
```

Para usar pesos e dados locais, monte os diretórios correspondentes no container e configure `YOLO_MODEL_PATH` e `DATABASE_PATH`.
