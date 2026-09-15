const $ = (selector) => document.querySelector(selector);
let audioContext;
let lastAlertKey = "";
let alarmTimer;
let alarmLevel = "";
let alertCooldown = 2.5;
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));

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
  const interval = Math.max(alertCooldown * 1000, level === "CRITICAL" ? 550 : level === "HIGH" ? 850 : 1300);
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
  $("#server-latency").textContent = state.metrics.server_detection_ms == null ? "-- ms" : `${state.metrics.server_detection_ms} ms`;
  $("#frame-drops").textContent = `${state.metrics.dropped_frames || 0} frames descartados`;
  $("#result-age").textContent = state.metrics.result_age_ms == null ? "Resultado: --" : `Idade do resultado: ${state.metrics.result_age_ms} ms`;
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
  $("#objects-table").innerHTML = state.detections.length ? state.detections.map((detection) => `<tr><td>#${detection.track_id}</td><td>${escapeHtml(detection.object_class.toUpperCase())}</td><td>${(detection.confidence * 100).toFixed(0)}%</td><td>${detection.proximity}%</td><td>${detection.approach_rate >= 0 ? "+" : ""}${detection.approach_rate}%/s</td><td>${pt(detection.position)}</td><td class="risk ${detection.risk}">${pt(detection.risk)}</td></tr>`).join("") : '<tr><td colspan="7" class="empty-row">Nenhuma detecção. Conecte a câmera do celular para começar.</td></tr>';
  $("#events").innerHTML = state.events.length ? state.events.slice(0, 12).map((event) => `<div class="event ${event.level}"><time>${new Date(event.timestamp).toLocaleTimeString()}</time><span>${escapeHtml(eventText(event.message))}</span></div>`).join("") : '<div class="empty-row">Os eventos da missão aparecerão aqui.</div>';
}

async function poll() {
  try { render(await (await fetch("/api/state")).json()); } catch (error) { console.error(error); }
  setTimeout(poll, 200);
}

async function init() {
  translateStaticInterface();
  connectProcessedVideo();
  const session = await (await fetch("/api/session")).json();
  $("#session-code").textContent = session.session;
  $("#phone-url").textContent = session.phone_url;
  $("#qr").src = "/api/qr";
  let configuration = await (await fetch("/api/settings")).json();
  const form = $("#settings-dialog form");
  const settingsDialog = $("#settings-dialog");
  const changed = new Set();
  const presets = await (await fetch("/api/presets")).json();
  function fillSettings(values) {
    configuration = values;
    Object.entries(values).forEach(([key, value]) => {
      const input = form.elements.namedItem(key);
      if (!input) return;
      if (input.type === "checkbox") input.checked = value;
      else input.value = Array.isArray(value) ? value.join(", ") : value;
    });
    settingsDialog.dataset.sound = String(values.sound_enabled);
    alertCooldown = values.alert_cooldown;
    $("#overlay-status").textContent = values.overlay_enabled ? "ON" : "OFF";
    changed.clear();
    stopContinuousAlert();
  }
  async function refreshModel() {
    const response = await fetch("/api/model");
    const model = await response.json();
    $("#loaded-model").textContent = (model.loaded_model || "Ainda não carregado") +
      (model.pending_reload ? ` · aguardando ${model.configured_model}` : "");
    $("#available-classes").textContent = `Disponíveis: ${model.available_classes.join(", ") || "aguardando carregamento"}` +
      (model.missing_classes.length ? `. Ausentes: ${model.missing_classes.join(", ")}` : "");
  }
  fillSettings(configuration);
  form.addEventListener("input", (event) => {
    if (event.target.name) changed.add(event.target.name);
  });
  // A newly selected preset supersedes earlier edits in its group.
  const presetFields = {
    performance_profile: ["image_size", "inference_fps", "half_precision"],
    proximity_preset: ["proximity_scale", "medium_threshold", "high_threshold", "critical_threshold", "approach_threshold"],
  };
  Object.entries(presetFields).forEach(([name, fields]) => {
    form.elements.namedItem(name).addEventListener("change", () => {
      fields.forEach((field) => changed.delete(field));
      const preset = presets[name][form.elements.namedItem(name).value];
      if (preset) Object.entries(preset).forEach(([field, value]) => {
        const input = form.elements.namedItem(field);
        if (input.type === "checkbox") input.checked = value;
        else input.value = value;
      });
      changed.add(name);
    });
  });
  $("#settings-button").addEventListener("click", async () => {
    settingsDialog.showModal();
    $("#settings-error").textContent = "";
    try {
      fillSettings(await (await fetch("/api/settings")).json());
      await refreshModel();
    } catch (error) {
      $("#settings-error").textContent = "Não foi possível carregar as configurações.";
    }
  });
  $("#close-settings").onclick = () => settingsDialog.close();
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const values = {};
    changed.forEach((name) => {
      const input = form.elements.namedItem(name);
      if (name === "monitored_classes") values[name] = input.value.split(",").map((value) => value.trim()).filter(Boolean);
      else if (input.type === "checkbox") values[name] = input.checked;
      else if (input.type === "number") values[name] = Number(input.value);
      else values[name] = input.value;
    });
    $("#save-settings").disabled = true;
    try {
      const response = await fetch("/api/settings", {
        method: "PATCH", headers: {"Content-Type": "application/json"}, body: JSON.stringify({values}),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(Array.isArray(body.detail) ? body.detail.map((item) => item.msg).join("; ") : body.detail);
      fillSettings(body);
      await refreshModel();
      $("#settings-error").textContent = "";
      settingsDialog.close();
    } catch (error) {
      $("#settings-error").textContent = `Não foi possível aplicar: ${error.message}`;
    } finally {
      $("#save-settings").disabled = false;
    }
  });
  document.querySelectorAll(".view-toggle button").forEach((button, index) => {
    button.onclick = async () => {
      const response = await fetch("/api/settings", {method:"PATCH", headers:{"Content-Type":"application/json"},
        body:JSON.stringify({values:{overlay_enabled:index === 0}})});
      if (response.ok) {
        fillSettings(await response.json());
        document.querySelectorAll(".view-toggle button").forEach((item) => item.classList.toggle("active", item === button));
      }
    };
  });
  $("#clear-events").onclick = () => { $("#events").innerHTML = ""; };
  poll();
}

init();
