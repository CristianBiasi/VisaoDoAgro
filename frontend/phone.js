let stream;
let peer;
let facing = "environment";
let rearCameras = [];
let cameraIndex = 0;
let selectedDeviceId = "";
const video = document.querySelector("#local-video");
const start = document.querySelector("#start-camera");
const status = document.querySelectorAll(".status-line");
document.title = "AgroSafe Vision | Câmera remota";
document.querySelector(".phone-title h1").textContent = "Transmita a visão do campo.";
document.querySelector(".phone-title p").textContent = "Mantenha esta página aberta enquanto o notebook analisa a câmera ao vivo.";
document.querySelector("#start-camera").textContent = "INICIAR CÂMERA TRASEIRA";
document.querySelector("#switch-camera").textContent = "TROCAR CÂMERA";
document.querySelector("#stop-camera").textContent = "ENCERRAR TRANSMISSÃO";

function setStatus(index, active, text) {
  status[index].classList.toggle("active", Boolean(active));
  status[index].innerHTML = `<i></i> ${text}`;
}

async function disconnectCamera() {
  stream?.getTracks().forEach((track) => track.stop());
  if (peer) {
    peer.close();
    peer = null;
    // Wait for the server worker before starting another camera/tracker session.
    await fetch("/api/camera/stop", {method: "POST"});
  }
}

async function connect() {
  try {
    await disconnectCamera();
    const videoConstraints = selectedDeviceId
      ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 1920 }, height: { ideal: 1080 }, aspectRatio: { ideal: 16 / 9 } }
      : { facingMode: { ideal: facing }, width: { ideal: 1920 }, height: { ideal: 1080 }, aspectRatio: { ideal: 16 / 9 } };
    stream = await navigator.mediaDevices.getUserMedia({
      video: videoConstraints,
      audio: false,
    });
    video.srcObject = stream;
    const activeTrack = stream.getVideoTracks()[0];
    selectedDeviceId = activeTrack.getSettings().deviceId || selectedDeviceId;
    if (facing === "environment") {
      const devices = await navigator.mediaDevices.enumerateDevices();
      rearCameras = devices.filter((device) => device.kind === "videoinput" && /back|rear|environment|wide|ultra|0\.5|0,5/i.test(device.label));
      const wideIndex = rearCameras.findIndex((device) => /ultra|wide|0\.5|0,5/i.test(device.label));
      if (wideIndex >= 0 && rearCameras[wideIndex].deviceId !== selectedDeviceId && !selectedDeviceId.includes("wide")) {
        cameraIndex = wideIndex;
        stream.getTracks().forEach((track) => track.stop());
        selectedDeviceId = rearCameras[wideIndex].deviceId;
        return connect();
      }
    }
    setStatus(0, true, "Câmera ativa");
    setStatus(1, false, "Conectando ao notebook");

    peer = new RTCPeerConnection();
    peer.ontrack = (event) => {
      const [remoteStream] = event.streams;
      if (remoteStream) {
        video.srcObject = remoteStream;
      }
    };
    peer.onconnectionstatechange = () => {
      if (!peer) return;
      const connected = ["connected", "completed"].includes(peer.connectionState);
      setStatus(1, connected, connected ? "Notebook conectado" : `Notebook ${peer.connectionState}`);
    };
    stream.getTracks().forEach((track) => peer.addTrack(track, stream));

    await peer.setLocalDescription(await peer.createOffer());
    const response = await fetch("/api/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sdp: peer.localDescription.sdp, type: peer.localDescription.type }),
    });
    if (!response.ok) throw new Error(`WebRTC server returned ${response.status}`);
    await peer.setRemoteDescription(await response.json());
    setStatus(1, true, "Notebook conectado");
    setStatus(2, "geolocation" in navigator, "GPS ativo");
  } catch (error) {
    setStatus(0, false, "Câmera indisponível");
    setStatus(1, false, "Falha na conexão com o notebook");
    console.error(error);
  }
}

start.onclick = connect;
document.querySelector("#switch-camera").onclick = async () => {
  stream?.getTracks().forEach((track) => track.stop());
  peer?.close();
  if (facing === "environment" && rearCameras.length > 1) {
    cameraIndex = (cameraIndex + 1) % rearCameras.length;
    selectedDeviceId = rearCameras[cameraIndex].deviceId;
  } else {
    facing = facing === "environment" ? "user" : "environment";
    selectedDeviceId = "";
  }
  await connect();
};
document.querySelector("#stop-camera").onclick = async () => {
  await disconnectCamera();
  video.srcObject = null;
  setStatus(0, false, "Câmera parada");
  setStatus(1, false, "Aguardando notebook");
};

if ("geolocation" in navigator) {
  navigator.geolocation.watchPosition((position) => {
    const coordinates = position.coords;
    fetch("/api/telemetry", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        latitude: coordinates.latitude,
        longitude: coordinates.longitude,
        altitude: coordinates.altitude,
        accuracy: coordinates.accuracy,
        speed_kmh: coordinates.speed == null ? null : coordinates.speed * 3.6,
        heading: coordinates.heading,
        timestamp: Date.now() / 1000,
        source: coordinates.speed == null ? "GPS ESTIMATED" : "GPS",
      }),
    }).catch(console.error);
  }, () => setStatus(2, false, "GPS indisponível"));
}
