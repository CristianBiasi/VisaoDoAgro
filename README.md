# AgroSafe Vision

MVP de monitoramento visual com FastAPI, WebRTC, vídeo offline, Ultralytics YOLO,
tracking, telemetria, banco de dados PostgreSQL, dashboard e alertas de risco.
A proximidade é uma estimativa visual: **não mede metros e não substitui um sistema de segurança certificado.**

## Documentação do código

Leia a [documentação completa do código](docs/CODIGO.md) para entender os arquivos,
classes, funções, API, fluxo WebRTC, cálculo de proximidade/risco, frontend,
configurações, treinamento e testes. Inclui diagramas e um mapa de manutenção.

## Executar no Windows

**Pré-requisito (Banco de Dados):**
A aplicação requer uma conexão com banco de dados PostgreSQL para persistir missões, telemetria, obstáculos e configurações.
Copie o arquivo `.env.example` para `.env` na raiz do projeto e defina a variável `BD` com a URL de conexão do seu banco (não versione o arquivo `.env`):

```powershell
copy .env.example .env
```

Instale Python 3.11+ e, na pasta do projeto, execute:

```powershell
.\start.bat
```

O launcher cria o ambiente e instala dependências. Abra o endereço HTTP/HTTPS
mostrado no terminal. Mantenha notebook e celular no mesmo Wi-Fi; escaneie o QR
Code e permita câmera e localização. No celular, o acesso à câmera requer HTTPS.
O certificado local é gerado por `generate_cert.py` usando cryptography.
Não requer OpenSSL/mkcert. Aceite o certificado local no navegador para a demonstração.

Se a pasta foi copiada de outro computador, a `.venv` antiga não é portável:
renomeie-a para backup e execute novamente o launcher.

Execução manual (com `uv` ou `pip`), depois de instalar as dependências:

```powershell
# Opção 1: Usando uv
uv sync
uv run python generate_cert.py
uv run uvicorn app:app --host 0.0.0.0 --port 8000 --ssl-keyfile .certs/key.pem --ssl-certfile .certs/cert.pem

# Opção 2: Usando venv + pip
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe generate_cert.py
.\.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000 --ssl-keyfile .certs/key.pem --ssl-certfile .certs/cert.pem
```

## Banco de Dados e Persistência

O projeto utiliza o **SQLAlchemy** integrado ao **PostgreSQL** (via Neon DB) para persistir as informações operacionais. Na primeira execução, as tabelas são criadas automaticamente se não existirem.

O que é salvo no banco:
- **Missões:** Histórico de sessões iniciadas e paradas via interface (`missoes`).
- **Alertas e Obstáculos:** Registros detalhados de objetos com risco alto ou crítico que apareceram no percurso, armazenados em `eventos_alerta` e `obstaculos_detectados`.
- **Telemetria:** Pontos de GPS e orientação gravados durante a missão (`telemetria_pontos`).
- **Configurações:** As alterações feitas no menu "Settings" são salvas na tabela `configuracoes` e sobrevivem à reinicialização.
- **Testes de Qualidade:** Ao rodar avaliações de IA no treinamento, os resultados (mAP, Precision, Recall, Falsos Positivos e Falsos Negativos) são inseridos na tabela `avaliacoes_modelo`.

## Arquitetura e fluxo

```text
celular (WebRTC) ──┐
                   ├→ recepção contínua → frame mais recente
vídeo local (MP4) ─┘                    ├→ inferência serial fora do event loop
                                        │  YOLO → tracking → proximidade → risco
                                        └→ vídeo atual + último overlay ainda válido
                                           ├→ WebRTC de volta ao celular
                                           └→ JPEG → dashboard
```

- `app.py`: rotas, sessão, telemetria, eventos, upload de vídeo e integração com BD.
- `video_pipeline.py`: slots de tamanho 1, descarte, métricas e ciclo de vida WebRTC.
- `local_video.py`: leitura e processamento assíncrono de vídeos offline enviados via dashboard.
- `detector.py`: carregamento YOLO (`.pt`, `.onnx`, `.engine`), classes e device (`cpu`, `cuda`, `directml`); retorna objetos estruturados.
- `processing.py`: compõe detecção, `ProximityEstimator` e `CollisionRiskEngine`.
- `overlay.py`: desenha sobre o frame atual, nas coordenadas originais.
- `config.py`: validação e presets, preservando a API de campos planos.
- `database.py`: modelos SQLAlchemy e sessão de persistência no PostgreSQL.
- `models.py`: contratos do dashboard e telemetria; `best.pt` / `yolo11n.pt` / `yolo11n.onnx`: pesos disponíveis.

A inferência bloqueante executa fora do event loop porque a fila de recepção do
aiortc continuaria acumulando frames enquanto o YOLO bloqueasse a rede.
Existe apenas uma inferência em execução e um frame aguardando, sempre substituído
pelo mais recente. A renderização também ocorre fora do event loop.
Nenhuma fila de inferências é criada. Cada dashboard tem um único JPEG pendente;
um espectador lento não bloqueia o recebimento nem os outros espectadores.

O runtime atende **uma câmera ou vídeo ativo por missão**. Uma segunda conexão WebRTC
recebe 409 para evitar misturar IDs de tracking e estado de risco. Encerre a câmera
atual antes de conectar outra. As tarefas são encerradas e a inferência em andamento
é aguardada antes de liberar o detector.

### Monitoramento com Vídeo Offline

Além da câmera ao vivo via WebRTC, o dashboard permite enviar arquivos de vídeo locais (`POST /api/video/upload`) para análise e simulação de missão:
- O `LocalVideoProcessor` (`local_video.py`) lê o vídeo com OpenCV em background, converte cada frame e executa o mesmo pipeline de detecção, estimativa de proximidade, motor de risco e persistência no banco.
- O envio de um vídeo encerra automaticamente qualquer stream de câmera ativo (e vice-versa), evitando conflito de IDs de rastreamento.
- O processamento pode ser interrompido a qualquer momento via `POST /api/video/stop`.

## Processar vídeo offline (GPU)

Para processar vídeos gravados de inspeção de torres e linhas de transmissão em alta qualidade, sem interface web e com taxa constante de 30 FPS, utilize o script `processar_video.py` ou o executável em lote `processar_video.bat`.

### Pré-requisitos e Aceleração por GPU (CUDA)

1. **Driver NVIDIA:** Certifique-se de que o driver NVIDIA esteja instalado e atualizado executando `nvidia-smi` no terminal.
2. **PyTorch com CUDA:** Para usar a GPU (ex.: NVIDIA GeForce RTX), instale o PyTorch com suporte a CUDA configurado no `pyproject.toml` ou com:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```
   Validação do CUDA:
   ```powershell
   .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Sem CUDA')"
   ```
   *Nota: Caso o CUDA não esteja disponível, o script executa automaticamente em CPU com aviso informativo.*

### Como Executar

- **Via duplo clique (Interface amigável):**
  Dê um duplo clique em `processar_video.bat`. O script abrirá uma janela nativa do Windows para você selecionar o vídeo de entrada. Ao finalizar o processamento com barra de progresso no terminal, abrirá a janela "Salvar como" para escolher onde salvar o vídeo resultante.
- **Via Linha de Comando (CLI para automação):**
  ```powershell
  # Processamento com caminhos definidos
  .\.venv\Scripts\python.exe processar_video.py --input caminho/video.mp4 --output caminho/resultado.mp4

  # Ajustando parâmetros (exemplo: inferência 640px, threshold 0.40)
  .\.venv\Scripts\python.exe processar_video.py --input video.mp4 --output saida.mp4 --conf 0.40 --batch 8
  ```

### Gerenciamento e Limites de Recursos

O script foi projetado para rodar em segundo plano sem travar o computador nem esgotar a memória:
- **CPU (~50%):** O script limita o uso a metade das threads lógicas da máquina (afinidade de CPU e limites OpenMP/MKL/OpenCV) e roda em prioridade de processo reduzida (`BELOW_NORMAL_PRIORITY_CLASS`).
- **RAM (máx. 10 GB):** Filas de frames desacopladas com controle estrito de memória e monitor em segundo plano que previne estouro de RAM (OOM).
- **Taxa Constante de 30 FPS:** O vídeo de saída é reamostrado de forma fluida para 30 FPS exatos mantendo a duração original do vídeo.
- **Áudio Original:** O áudio é preservado sincronizado na saída (usando FFmpeg embutido via `imageio-ffmpeg`).

### Relatório de Métricas (`.json`)

Ao finalizar, um arquivo `<saida>.json` é gerado automaticamente com estatísticas da execução:
- `entrada` e `saida`: caminhos dos arquivos.
- `dispositivo`: hardware utilizado (ex.: `cuda:0 (NVIDIA RTX 2060 SUPER, FP16)` ou `cpu`).
- `resolucao`, `fps_entrada`, `fps_saida`: especificações de vídeo.
- `frames_processados`, `duracao_video_s`: total de quadros e duração do vídeo.
- `tempo_processamento_s`, `velocidade_media_fps`: tempo total e velocidade média de inferência.
- `pico_ram_gb`, `pico_vram_gb`, `cpu_media_pct`: consumo máximo de memória e média de processamento.
- `deteccoes_por_classe`: contagem acumulada de instâncias detectadas (`Cabo`, `Torre treliçada`, etc.).
- `frames_com_deteccao_por_classe`: total de quadros em que cada classe esteve presente.
- `confianca_media_por_classe`: score médio de confiança das predições por classe.

## Performance profiles

BALANCED é o padrão. Estes valores são limites/configurações, não FPS garantidos.

| Perfil | Tamanho YOLO | Limite IA/s | FP16 solicitado |
|---|---:|---:|---|
| QUALITY | 640 | 12 | não |
| BALANCED | 512 | 15 | não |
| PERFORMANCE | 416 | 24 | sim, somente CUDA |

Tracking permanece ligado nos três perfis. O perfil PERFORMANCE reduz explicitamente
o tamanho de entrada; pode perder detalhes de objetos pequenos. BALANCED aumenta de
416 para 512 em relação ao projeto anterior. Não há resize intermediário fixo em 640:
o Ultralytics faz seu próprio pré-processamento mantendo as caixas na imagem original.

Preservamos o tracker padrão do Ultralytics; trocar por um mais simples exige avaliação
de estabilidade dos IDs. `lap`, usado pelo tracking, é instalado antecipadamente para
evitar instalação no primeiro frame. Todas as caixas são transferidas para CPU juntas,
em vez de várias transferências por objeto. JPEG só é codificado quando há clientes
e respeita o limite de 12/s. O vídeo não fica congelado no último frame da IA.

### CPU/GPU (NVIDIA CUDA e AMD Radeon DirectML)

`device=auto` escolhe CUDA quando o PyTorch instalado a disponibiliza; caso contrário,
usa CPU. Advanced aceita `cpu`, `cuda`, `cuda:N` e `directml` (ou `dml`).
- **NVIDIA (CUDA):** FP16 só é solicitado para CUDA; falha de runtime CUDA tenta novamente em CPU/FP32 e registra o erro. Um build CPU do PyTorch não passa a usar a GPU apenas por alterar esse campo.
- **AMD Radeon (DirectML):** Ao utilizar um modelo `.onnx` (como `yolo11n.onnx`) com `device=directml`, o detector inicializa o `onnxruntime-directml` (`DmlExecutionProvider`), permitindo aceleração por GPU em placas AMD no Windows (com fallback automático para CPU caso o provedor não esteja disponível).

### Métricas e avaliação

- **inference_ms**: execução YOLO/tracking e extração das caixas, sem carga inicial.
- **AI FPS**: conclusões efetivas em janela móvel de 2 segundos, incluindo o limitador.
- **video FPS**: frames entregues à trilha de saída; não é FPS de captura do celular.
- **server_detection_ms**: frame recebido pela aplicação → resultado disponível,
  incluindo espera, conversão e eventual carregamento inicial.
- **result_age_ms**: idade do frame que originou o último resultado.
- **processing_ms**, **overlay_ms**, **jpeg_ms**, **dropped_frames**, device/backend/FP16:
  disponíveis em `/api/state`. Overlay inclui conversão para BGR.
- A latência de servidor **não** inclui captura, rede de entrada, decodificação anterior
  ao recv, rede de saída ou apresentação no navegador. Não é end-to-end.

Resultados com mais de 1 segundo deixam de produzir overlay/alerta ativo. Um ajuste
de configuração também invalida o resultado antigo até a próxima inferência.
A aplicação consome a fila pública de recebimento continuamente, mas não controla
buffers do navegador, codec ou rede. Não promete eliminar toda fonte de atraso.

Não há ganho percentual de desempenho comprovado. Compare os perfis com a mesma
câmera, iluminação, resolução, cena e máquina, depois do aquecimento do modelo.
Para medir câmera → tela, filme simultaneamente um evento visual e a tela com um
relógio comum; os relógios RTP e do servidor não são diretamente comparáveis.

## Sensibilidade de proximidade

No dashboard: **Próxima / Normal / Distante**. Distante alerta mais cedo; Próxima,
mais tarde. Ajustam escala visual e limiares de risco/aproximação. Normal mantém
os valores anteriores. Não são distâncias físicas configuradas.

A área relativa da bounding box, suavizada no tempo, produz um score de 0 a 100.
A variação desse score produz aproximação em pontos/s. Tamanho real do objeto,
zoom, lente, perspectiva e oclusão alteram a estimativa. Sem calibração e/ou
profundidade, não se deve apresentar esse score como metros.

Advanced permite ajustar individualmente escala, suavização, limiares, corredor
e confirmação. Os limiares devem respeitar médio < alto < crítico.
O modo predict sem tracking não reutiliza IDs artificiais entre frames: não calcula
aproximação temporal nem confirmação entre objetos que podem ser diferentes.

`DetectionProcessor` recebe opcionalmente um provedor de proximidade com
`update(...)` e `forget_missing(...)`. Essa é a fronteira simples para uma futura
fonte de profundidade. Sensores estéreo, LiDAR e drone não foram implementados;
uma distância física futura precisará de um campo com unidade e origem próprios.

## Configurações ao vivo

`GET /api/settings` retorna os campos atuais.
`PATCH /api/settings` recebe `{"values": {...}}` e retorna a configuração validada.
Campos desconhecidos, tamanhos inválidos e limiares inconsistentes retornam 422,
sem modificar a configuração anterior.

Os presets são aplicados antes de overrides explícitos no mesmo pedido.
Valores avançados que divergem de um preset identificam o perfil como CUSTOM.
Mudar um preset mantém os campos que ele não controla.
As configurações são mantidas em memória durante a execução e **persistidas automaticamente no banco de dados PostgreSQL** (tabela `configuracoes`), sendo recarregadas ao iniciar a aplicação.

`model_path`, device e precisão provocam recarga serial na próxima inferência.
Resultados calculados com a configuração anterior são descartados.
Trocar tracking recarrega o modelo para remover callbacks de rastreamento; classes reinicializam o tracker; mudanças de segurança limpam o histórico.
Falha ao carregar um modelo é registrada, não derruba a API e não produz resultados
silenciosamente usando o modelo anterior. Corrija o caminho nas configurações.

## Usando modelos YOLO customizados e ONNX

O repositório já inclui na raiz o modelo customizado treinado `best.pt`, além dos modelos base `yolo11n.pt` e `yolo11n.onnx`. Para usar o modelo treinado ou outro peso customizado (`.pt`, `.onnx` ou `.engine`):

1. Coloque seus pesos na raiz (ex.: `best.pt`) ou em `models/best.pt`.
2. Em Advanced Settings, altere o caminho de `model_path` (ex.: `best.pt` ou `yolo11n.onnx`).
3. Configure classes ou deixe a lista vazia para monitorar todas.
4. A próxima inferência carrega o modelo; reabra SETTINGS para consultar as classes.

Exemplo da API:

```json
{
  "values": {
    "model_path": "best.pt",
    "monitored_classes": ["fio", "poste", "torre", "construcao", "passaro"],
    "confidence": 0.45,
    "iou": 0.45
  }
}
```

Os nomes são obtidos de `model.names`; não assumimos índices COCO. Classes configuradas
ausentes são ignoradas com aviso. Se nenhuma classe configurada existir, retorna zero
detecções; lista vazia é a escolha explícita para todas. `GET /api/model` informa
modelo configurado/carregado, classes disponíveis/ausentes e recarga pendente.

O modelo genérico `yolo11n.pt` continua na raiz por compatibilidade e pode ser baixado
pelo Ultralytics caso ausente. Modelos customizados precisam existir localmente.
Caminhos relativos são resolvidos a partir da raiz do projeto.
Formatos `.pt`, `.onnx` e `.engine` são aceitos pela validação de configuração.
O treinamento offline está organizado em `training/`, separado do runtime.

## Treinar fios, postes, torres, construções e pássaros

As pastas `datasets/obstaculos/images/{train,val,test}` e
`datasets/obstaculos/labels/{train,val,test}` estão preparadas para o dataset YOLO,
com as cinco classes definidas em `datasets/obstaculos/data.yaml`. Para converter anotações do dataset TTPLA (polígonos JSON) para o formato YOLO, utilize o script `convert_ttpla_to_yolo.py`.

Depois de adicionar imagens e anotações YOLO:

```powershell
.\treinar.bat --check
.\treinar.bat
```

O script `training/train.py` executa o treinamento, gera relatórios detalhados de qualidade (mAP, Precision, Recall, Falsos Positivos e Falsos Negativos por classe) e salva os resultados na tabela `avaliacoes_modelo` do banco de dados. Leia o [passo a passo de treinamento](training/README.md) para mais detalhes.

## Testes

Consulte [os cinco testes unitários mais importantes](docs/TESTES_UNITARIOS.md)
para entender os cenários, resultados esperados, isolamento do banco e como executar.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

Inclui os testes originais e cenários de presets, validação, histerese,
limpeza de histórico, seleção de device, FP16/fallback, recarga de modelo,
classes customizadas, transferência única de caixas, frames descartados,
timestamps de vídeo e troca de configuração durante inferência.
O detector é simulado: a suíte não baixa pesos e não requer GPU.

## Limitações

- O modelo genérico (`yolo11n.pt`) não é especializado em fios, postes ou cercas agrícolas; para esse cenário, utilize o modelo treinado (`best.pt`).
- Tracking/risco com câmera móvel precisam de avaliação em cenas reais.
- O primeiro carregamento/aquecimento custa mais que inferências posteriores.
- Não há distância física RGB em metros (apenas estimativa visual), integração MAVLink ou sensores de profundidade dedicados (LiDAR/estéreo).
- Certificado local, firewall e Wi-Fi precisam permitir acesso pelo celular.
- A confirmação é por inferência; o tempo até confirmar varia com o desempenho.
- Áudio depende de interação no navegador e volume do dispositivo.

Referências de integração:
[Ultralytics predict](https://docs.ultralytics.com/modes/predict/) e
[Ultralytics tracking](https://docs.ultralytics.com/modes/track/).
