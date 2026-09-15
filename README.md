# AgroSafe Vision

MVP funcional de prevenção de colisões para demonstração com smartphone, notebook e YOLO. O notebook executa o servidor FastAPI, recebe a câmera do telefone por WebRTC, processa os frames com Ultralytics YOLO e entrega um dashboard de monitoramento em tempo real.

## Funcionalidades

- `/` dashboard HUD responsivo para notebook.
- `/phone` cliente simples para câmera traseira do celular.
- WebRTC com `aiortc`: o servidor recebe vídeo e devolve a trilha processada.
- YOLO real lazy-load, usando `yolo11n.pt` por padrão e baixando o modelo na primeira detecção.
- Tracking persistente do Ultralytics quando habilitado.
- Proximity Score calculado pela área da bounding box, sem inventar metros.
- Approach Rate temporal, corredor central e risco SAFE/LOW/MEDIUM/HIGH/CRITICAL.
- Histerese, confirmação de frames, QR Code, GPS, eventos, métricas e configurações ao vivo.

## Executar no Windows

Na pasta do projeto:

```powershell
.\start.bat
```

Ou manualmente:

```powershell
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Abra `https://localhost:8000` quando o launcher indicar HTTPS. O endereço LAN aparece no launcher. Para acessar a câmera de outro dispositivo, aceite o aviso do certificado local no celular e continue para a página. O notebook e o smartphone devem estar na mesma rede Wi-Fi.

O notebook e o smartphone devem estar na mesma rede Wi-Fi. No dashboard, escaneie o QR Code. No telefone, permita a câmera traseira e a localização. O vídeo só passa a ser processado quando houver um stream real.

## Demonstração

1. Inicie o launcher e abra o dashboard.
2. Escaneie o QR Code com o telefone.
3. Pressione `START REAR CAMERA`.
4. Aponte para uma cadeira ou pessoa.
5. Aproxime-se e observe a caixa, `PROXIMITY`, `APPROACH` e o risco.
6. Ajuste os thresholds em `SETTINGS`; a API aplica mudanças sem reiniciar.
7. O áudio depende da interação inicial do navegador e do volume do sistema.

## Configuração e modelo

As configurações vivem em `config.py` e podem ser alteradas pela API `/api/settings`. O campo `model_path` aceita um futuro `best.pt`; não há nomes de modelo espalhados pelo processamento. As classes monitoradas são filtradas antes da inferência.

A proximidade é `VISION / CALCULATED`: uma indicação relativa baseada no tamanho e crescimento do objeto na imagem. Não é distância física. Velocidade, GPS e heading são `GPS` quando disponíveis; caso contrário permanecem `UNAVAILABLE`.

## Testes

```powershell
python -m unittest tests.test_logic -v
```

## Limitações e roadmap

O modelo genérico não conhece fios, postes, cercas ou obstáculos agrícolas específicos. Treine e informe um `best.pt` para isso. HTTPS automático depende de OpenSSL/mkcert instalado no Windows. Não há ainda medição RGB em metros, MAVLink, mapa persistente ou gravação de vídeo.

As fronteiras atuais (`YoloDetector`, `ProximityEstimator`, `CollisionRiskEngine`, telemetria e câmera WebRTC) permitem adicionar `DistanceProvider`, `MavlinkTelemetryProvider`, RTSP/USB e persistência SQLite sem alterar o dashboard principal.
