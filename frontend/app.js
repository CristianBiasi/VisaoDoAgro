const $ = (selector) => document.querySelector(selector);
let audioContext;
let lastAlertKey = "";
let alarmTimer;
let alarmLevel = "";

const pt = (value) => ({
  SAFE: "SEGURO", LOW: "BAIXO", MEDIUM: "MÉDIO", HIGH: "ALTO", CRITICAL: "CRÍTICO",
  LEFT: "ESQUERDA", CENTER: "CENTRO", RIGHT: "DIREITA",
  "STREAM CONNECTED": "TRANSMISSÃO ATIVA", "STREAM OFFLINE": "TRANSMISSÃO OFFLINE",
  "GPS ACTIVE": "GPS ATIVO", "GPS OFFLINE": "GPS OFFLINE", "AI DEVICE": "DISPOSITIVO IA",
}[value] || value);

function translateStaticInterface() {
  const translations = {
    "LIVE MISSION": "MISSÃO AO VIVO", "COLLISION AWARENESS SYSTEM": "SISTEMA DE PREVENÇÃO DE COLISÕES",
    "VISION ENGINE / REALTIME": "MOTOR DE VISÃO / TEMPO REAL", "Mission monitor": "Monitor da missão",
    "AI VIEW": "VISÃO IA", RAW: "ORIGINAL", "PAIR DEVICE": "CONECTAR DISPOSITIVO", "SESSION READY": "SESSÃO PRONTA",
    "Connect camera": "Conectar câmera", "Scan with the rear camera phone to join this mission.": "Escaneie com o celular para entrar nesta missão.",
    "PRIMARY THREAT": "AMEAÇA PRINCIPAL", "No active threat": "Nenhuma ameaça ativa", "Objects in the corridor will appear here": "Objetos no corredor aparecerão aqui",
    TELEMETRY: "TELEMETRIA", SPEED: "VELOCIDADE", HEADING: "DIREÇÃO", MISSION: "MISSÃO", OBJECTS: "OBJETOS",
    TRACKED: "RASTREADOS", "TRACKED OBJECTS": "OBJETOS RASTREADOS", "EVENT LOG": "REGISTRO DE EVENTOS", CLEAR: "LIMPAR",
    "No detections. Connect a phone camera to begin.": "Nenhuma detecção. Conecte a câmera do celular para começar.",
    "Mission events will appear here.": "Os eventos da missão aparecerão aqui.", "COLLISION CORRIDOR": "CORREDOR DE COLISÃO",
    "ACTIVE · CENTER BIAS": "ATIVO · FOCO CENTRAL", "LIVE CONFIGURATION": "CONFIGURAÇÃO AO VIVO",
    "Confidence threshold": "Limite de confiança", "Corridor width": "Largura do corredor", "Corridor height": "Altura do corredor",
    "Medium risk threshold": "Limite de risco médio", "High risk threshold": "Limite de risco alto", "Critical risk threshold": "Limite de risco crítico",
    "Approach threshold": "Limite de aproximação", "Sound alerts enabled": "Alertas sonoros ativados", "APPLY LIVE": "APLICAR AGORA",
    "SETTINGS": "CONFIGURAÇÕES", "VIDEO FPS": "FPS DO VÍDEO", "AI FPS": "FPS DA IA", INFERENCE: "INFERÊNCIA",
    "AI DEVICE": "DISPOSITIVO IA", RESOLUTION: "RESOLUÇÃO", "PROXIMITY": "PROXIMIDADE", APPROACH: "APROXIMAÇÃO", POSITION: "POSIÇÃO", RISK: "RISCO",
  };
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    let text = walker.currentNode.nodeValue;
    Object.entries(translations).forEach(([from, to]) => { text = text.replaceAll(from, to); });
    walker.currentNode.nodeValue = text;
  }
}

function connectProcessedVideo() {
  const oldVideo = $("#remote-video");
  const image = document.createElement("img");
  image.id = "remote-video";
  image.alt = "Vídeo processado da câmera do telefone";
  image.style.cssText = "width:100%;height:100%;object-fit:contain;display:none";
  oldVideo.replaceWith(image);
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/ws/video`);
  socket.binaryType = "blob";
  socket.onmessage = (event) => {
    const previous = image.src;
    image.src = URL.createObjectURL(event.data);
    image.style.display = "block";
    if (previous) URL.revokeObjectURL(previous);
    $(".video-shell").classList.add("connected");
  };
  socket.onclose = () => setTimeout(connectProcessedVideo, 1500);
}

function riskClass(risk) { return (risk || "SAFE").toLowerCase(); }

function eventText(message) {
  return message.replace("Settings updated", "Configurações atualizadas").replace("Mission started", "Missão iniciada").replace("Mission stopped", "Missão encerrada").replace(" risk ", " risco ").replace("AI unavailable", "IA indisponível");
}

function beep(level) {
  if ($("#settings-dialog").dataset.sound === "false" || level === "SAFE" || level === "LOW") return;
  audioContext ||= new AudioContext();
  const oscillator = audioContext.createOscillator();
  const gain = audioContext.createGain();
  oscillator.frequency.value = level === "CRITICAL" ? 760 : 540;
  gain.gain.value = 0.06;
  oscillator.connect(gain);
  gain.connect(audioContext.destination);
  oscillator.start();
  oscillator.stop(audioContext.currentTime + (level === "CRITICAL" ? 0.45 : 0.16));
}

function stopContinuousAlert() {
  if (alarmTimer) clearInterval(alarmTimer);
  alarmTimer = undefined;
  alarmLevel = "";
}

function startContinuousAlert(level) {
  if (!["MEDIUM", "HIGH", "CRITICAL"].includes(level)) {
    stopContinuousAlert();
    return;
  }
  if (alarmLevel === level && alarmTimer) return;
  stopContinuousAlert();
  alarmLevel = level;
  beep(level);
  const interval = level === "CRITICAL" ? 550 : level === "HIGH" ? 850 : 1300;
  alarmTimer = setInterval(() => beep(alarmLevel), interval);
}

function render(state) {
  const status = state.status;
  const labels = [["PHONE", status.phone_connected], ["CAMERA", status.camera_connected], ["STREAM", status.stream_connected], ["YOLO", state.metrics.device !== "UNAVAILABLE"], ["GPS", status.gps_active], ["SENSORS", status.sensors_active]];
  $("#system-status").innerHTML = labels.map(([name, active]) => `<span><i style="color:${active ? "var(--mint)" : "var(--red)"}">●</i>${name}</span>`).join("");
  $(".video-shell").classList.toggle("connected", status.stream_connected);
  $("#stream-label").textContent = pt(status.stream_connected ? "STREAM CONNECTED" : "STREAM OFFLINE");
  $("#resolution").textContent = state.metrics.resolution;
  $("#video-fps").textContent = state.metrics.video_fps || "--";
  $("#ai-fps").textContent = state.metrics.ai_fps || "--";
  $("#inference").textContent = state.metrics.inference_ms ? `${state.metrics.inference_ms} ms` : "-- ms";
  $("#device").textContent = state.metrics.device;
  $("#speed").textContent = state.telemetry?.speed_kmh != null ? `${state.telemetry.speed_kmh.toFixed(1)} km/h` : "--";
  $("#heading").textContent = state.telemetry?.heading != null ? `${state.telemetry.heading.toFixed(0)}°` : "--";
  $("#gps-state").textContent = pt(status.gps_active ? "GPS ACTIVE" : "GPS OFFLINE");
  const minutes = Math.floor(state.metrics.mission_time / 60).toString().padStart(2, "0");
  const seconds = (state.metrics.mission_time % 60).toString().padStart(2, "0");
  $("#mission-time").textContent = `${minutes}:${seconds}`;
  $("#objects").textContent = state.metrics.objects;
  $("#object-count").textContent = `${state.metrics.objects} OBJECTS`;

  const threat = state.primary_threat;
  $("#threat-empty").hidden = Boolean(threat);
  $("#threat-data").hidden = !threat;
  $("#threat-risk").textContent = pt(threat?.risk || "SAFE");
  $("#threat-risk").className = `risk-tag ${riskClass(threat?.risk)}`;
  startContinuousAlert(threat?.risk || "SAFE");
  if (threat) {
    $("#threat-name").textContent = `${threat.object_class.toUpperCase()} #${threat.track_id}`;
    $("#threat-proximity").textContent = `${threat.proximity}%`;
    $("#threat-approach").textContent = `${threat.approach_rate >= 0 ? "+" : ""}${threat.approach_rate}%/s`;
    $("#threat-position").textContent = pt(threat.position);
    const alertKey = `${threat.track_id}:${threat.risk}`;
    if (alertKey !== lastAlertKey) lastAlertKey = alertKey;
  }
  $("#objects-table").innerHTML = state.detections.length ? state.detections.map((detection) => `<tr><td>#${detection.track_id}</td><td>${detection.object_class.toUpperCase()}</td><td>${(detection.confidence * 100).toFixed(0)}%</td><td>${detection.proximity}%</td><td>${detection.approach_rate >= 0 ? "+" : ""}${detection.approach_rate}%/s</td><td>${pt(detection.position)}</td><td class="risk ${detection.risk}">${pt(detection.risk)}</td></tr>`).join("") : '<tr><td colspan="7" class="empty-row">Nenhuma detecção. Conecte a câmera do celular para começar.</td></tr>';
  $("#events").innerHTML = state.events.length ? state.events.slice(0, 12).map((event) => `<div class="event ${event.level}"><time>${new Date(event.timestamp).toLocaleTimeString()}</time><span>${eventText(event.message)}</span></div>`).join("") : '<div class="empty-row">Os eventos da missão aparecerão aqui.</div>';
}

async function poll() {
  try { render(await (await fetch("/api/state")).json()); } catch (error) { console.error(error); }
  setTimeout(poll, 500);
}

async function init() {
  translateStaticInterface();
  connectProcessedVideo();
  const session = await (await fetch("/api/session")).json();
  $("#session-code").textContent = session.session;
  $("#phone-url").textContent = session.phone_url;
  $("#qr").src = "/api/qr";
  const configuration = await (await fetch("/api/settings")).json();
  const form = $("#settings-dialog form");
  form.insertAdjacentHTML("afterbegin", '<p class="settings-note">A proximidade é um score visual relativo, não representa metros.</p><label>Sensibilidade da proximidade <input name="proximity_scale" type="range" min="50" max="300" step="5"><output data-for="proximity_scale"></output></label><label>Suavização temporal <input name="proximity_smoothing" type="range" min=".05" max="1" step=".05"><output data-for="proximity_smoothing"></output></label>');
  Object.entries(configuration).forEach(([key, value]) => { const input = form.elements[key]; if (input) input.type === "checkbox" ? input.checked = value : input.value = value; });
  form.querySelectorAll("input[type=range]").forEach((input) => { const output = form.querySelector(`[data-for="${input.name}"]`); const update = () => { output.value = input.value; }; input.addEventListener("input", update); update(); });
  const settingsDialog = $("#settings-dialog");
  settingsDialog.dataset.sound = configuration.sound_enabled;
  $("#settings-button").addEventListener("click", () => {
    if (typeof settingsDialog.showModal === "function") settingsDialog.showModal();
    else settingsDialog.setAttribute("open", "");
  });
  $("#save-settings").onclick = async () => {
    const values = {};
    Array.from(form.elements).forEach((input) => { if (input.name) values[input.name] = input.type === "checkbox" ? input.checked : Number(input.value); });
    await fetch("/api/settings", { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ values }) });
    settingsDialog.dataset.sound = values.sound_enabled;
    if (!values.sound_enabled) stopContinuousAlert();
  };
  $("#clear-events").onclick = () => { $("#events").innerHTML = ""; };
  poll();
}

init();
