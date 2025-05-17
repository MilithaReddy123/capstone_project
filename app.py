import streamlit as st
import av
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from deepface import DeepFace
from liveness import is_real_face
import asyncio
import sys

# Fix for Windows event loop
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

st.set_page_config(page_title="Fast Emotion Detector", layout="wide")
st.title("⚡ Low-Lag Real-time Emotion Detection with Anti-Spoofing")

# WebRTC config to connect media stream
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Define the main video processor
class EmotionLivenessProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.result_emotion = "Initializing..."
        self.result_score = 0.0
        self.live_result = True

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        display_img = img.copy()

        # Process every 10th frame to reduce lag
        if self.frame_count % 10 == 0:
            try:
                resized_img = cv2.resize(img, (224, 224))  # Smaller for faster processing

                # Real liveness detection using eye detection
                self.live_result = is_real_face(resized_img)

                if self.live_result:
                    result = DeepFace.analyze(
                        resized_img,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv'
                    )
                    self.result_emotion = result[0]['dominant_emotion']
                    self.result_score = result[0]['emotion'][self.result_emotion]
                else:
                    self.result_emotion = "Fake Face"
                    self.result_score = 0.0

            except Exception as e:
                print("[ERROR]", e)
                self.result_emotion = "Error"
                self.result_score = 0.0

        label = f"{self.result_emotion} ({self.result_score:.1f}%)"
        color = (0, 255, 0) if self.live_result else (0, 0, 255)
        label += " | ✅ Real" if self.live_result else " | ❌ Fake"

        cv2.putText(display_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(display_img, format="bgr24")

# Start video streaming in Streamlit
webrtc_streamer(
    key="live-emotion-detector",
    video_processor_factory=EmotionLivenessProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)
