"""Local WebRTC round trip with synthetic frames and a mocked detector."""
import asyncio
import unittest
from unittest.mock import Mock, patch

from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack

import app


class WebRTCTests(unittest.IsolatedAsyncioTestCase):
    async def test_offer_returns_processed_video_and_cleans_up(self):
        client = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        detector = Mock()
        detector.device = "cpu"
        detector.half = False
        detector.last_inference = 1.0
        detector.detect.return_value = []
        received = asyncio.get_running_loop().create_future()

        @client.on("track")
        async def on_track(track):
            try:
                video_frame = await track.recv()
                if not received.done():
                    received.set_result(video_frame)
            except Exception as exc:
                if not received.done():
                    received.set_exception(exc)

        original = app.detector
        app.detector = detector
        try:
            client.addTrack(VideoStreamTrack())
            await client.setLocalDescription(await client.createOffer())

            class Request:
                async def json(self):
                    return {"sdp": client.localDescription.sdp, "type": client.localDescription.type}

            with patch.object(app, "RTCPeerConnection", side_effect=lambda: RTCPeerConnection(RTCConfiguration(iceServers=[]))):
                answer = await app.offer(Request())
            self.assertIsInstance(answer, dict)
            await client.setRemoteDescription(RTCSessionDescription(**answer))
            result = await asyncio.wait_for(received, timeout=15)
            self.assertGreater(result.width, 0)
            self.assertIsNotNone(result.pts)
            self.assertTrue(detector.detect.called)
            active = list(app.tracks.values())
            duplicate = await app.offer(Request())
            self.assertEqual(duplicate.status_code, 409)
            self.assertEqual(await app.stop_camera(), {"ok": True})
            self.assertFalse(app.tracks)
        finally:
            await client.close()
            await asyncio.gather(*(app.close_peer(pc) for pc in tuple(app.state["pcs"])))
            app.detector = original
        self.assertFalse(app.tracks)
        self.assertFalse(app.state["pcs"])
        self.assertTrue(all(track._infer_task.done() for track in active))
