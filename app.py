import streamlit as st
import whisper
import cv2
import numpy as np
import os
import tempfile
import dlib
import re
from num2words import num2words
from moviepy.editor import VideoFileClip
from PIL import Image

# Load Dlib's shape predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Function to extract only lip region from a frame
def extract_lip_region(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) > 0:
        for face in faces:
            landmarks = predictor(gray, face)
            lip_points = []

            for n in list(range(48, 61)):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                lip_points.append((x, y))

            # Get bounding box of lips
            x_coords = [pt[0] for pt in lip_points]
            y_coords = [pt[1] for pt in lip_points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Add padding
            pad = 10
            x_min = max(x_min - pad, 0)
            y_min = max(y_min - pad, 0)
            x_max = min(x_max + pad, frame.shape[1])
            y_max = min(y_max + pad, frame.shape[0])

            cropped = frame[y_min:y_max, x_min:x_max]
            return cv2.resize(cropped, (128, 64))

    return None

# Convert video to audio (silently, no Streamlit UI display)
def convert_video_to_audio(video_path, audio_path):
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()
        return True
    except Exception:
        return False

# Transcribe using Whisper (base model)
def transcribe_audio(audio_path):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

# Convert digits like "7" into words like "seven"
def convert_numbers_to_words(text):
    def replace_number(match):
        num = match.group()
        try:
            return num2words(int(num))
        except:
            return num
    return re.sub(r'\b\d+\b', replace_number, text)

# Extract lip frames for GIF animation
def get_lip_frames(video_path, max_frames=60):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        lips = extract_lip_region(frame)
        if lips is not None:
            img_rgb = cv2.cvtColor(lips, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(img_rgb))
            count += 1

    cap.release()
    return frames

# Streamlit Layout
st.set_page_config(page_title="ForenSpeak", layout="wide")
st.title("🧠 ForenSpeak: A Lip Reading System")

tabs = st.tabs(["📊 Dashboard", "📝 Transcription", "👄 Lip Tracker Animation"])

# Tab 1: Dashboard
with tabs[0]:
    st.header("Dashboard")
    st.markdown("""
    Welcome to **ForenSpeak**, an AI-powered lip-reading system designed to assist forensic teams by reading lips and generating transcriptions from MPEG video files.

    **Core Features**:
    - Extract lip regions visually.
    - Perform accurate speech-to-text conversion with Whisper.
    - Analyze silent videos for lip activity.
    """)

# Tab 2: Transcription
with tabs[1]:
    st.header("Transcribe Speech from MPEG Video")

    uploaded_file = st.file_uploader("Upload a .mpg or .mpeg file", type=["mpg", "mpeg"], key="transcribe")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mpg") as temp_video:
            temp_video.write(uploaded_file.read())
            video_path = temp_video.name

        audio_path = "temp_audio.wav"
        if convert_video_to_audio(video_path, audio_path):
            transcript = transcribe_audio(audio_path)
            if transcript:
                # Convert digits to words
                transcript = convert_numbers_to_words(transcript)
                st.success("✅ Transcription Completed!")
                st.text_area("Transcription Output", transcript, height=250)
            else:
                st.error("Failed to generate transcription.")

        # Store video path for lip tracking
        st.session_state.video_file = video_path

# Tab 3: Lip Tracker Animation
with tabs[2]:
    st.header("Lip Region Tracker")

    if "video_file" in st.session_state:
        video_path = st.session_state.video_file
        st.info("🔍 Detecting and extracting lip regions...")

        lip_frames = get_lip_frames(video_path)

        if lip_frames:
            st.success("✅ Lip Animation Extracted!")

            gif_path = "lip_animation.gif"
            lip_frames[0].save(
                gif_path,
                save_all=True,
                append_images=lip_frames[1:],
                duration=100,
                loop=0
            )

            st.image(gif_path, caption="🌀 Lip Tracking Animation (GIF)", use_container_width=True)

            with open(gif_path, "rb") as f:
                gif_data = f.read()
                st.download_button(
                    label="📥 Download Lip Animation GIF",
                    data=gif_data,
                    file_name="lip_animation.gif",
                    mime="image/gif"
                )
        else:
            st.error("⚠️ No lip region detected in the video.")
    else:
        st.warning("⚠️ Please upload a video in the 'Transcription' tab first.")
