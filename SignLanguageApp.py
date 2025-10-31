import streamlit as st
import tempfile
import cv2
import os
from ultralytics import YOLO
from gtts import gTTS
from googletrans import Translator

# ------------------ Load Model ------------------
model = YOLO("models/best.pt")
translator = Translator()

# ------------------ Streamlit Config ------------------
st.set_page_config(page_title="Sign2Text", page_icon="🖐️", layout="wide")
st.title("🖐️ Sign2Text - Sign Language Detection")
st.markdown("**Detect signs from webcam or uploaded video, translate them, and listen to audio. 🎧**")

# ------------------ Language Options ------------------
languages = {
    "Tamil": "ta",
    "Malayalam": "ml",
    "Japanese": "ja",
    "Hindi": "hi",
    "German": "de",
    "Chinese": "zh-cn"
}

# ------------------ Sidebar ------------------
st.sidebar.title("Settings")
mode = st.sidebar.radio("📌 Select Mode", ["Open Webcam", "Upload Video"])
lang_choice = st.sidebar.selectbox("🌐 Translation Language", list(languages.keys()))
target_lang = languages[lang_choice]

# ------------------ Session State ------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "latest" not in st.session_state:
    st.session_state.latest = None
if "run_webcam" not in st.session_state:
    st.session_state.run_webcam = False
if "run_video" not in st.session_state:
    st.session_state.run_video = False

# ------------------ Show Prediction ------------------
def show_prediction(word, target_lang, lang_choice):
    translated_word = translator.translate(word, dest=target_lang).text

    # Generate audios
    en_audio_path = "temp_en.mp3"
    gTTS(word, lang="en").save(en_audio_path)
    trans_audio_path = "temp_trans.mp3"
    gTTS(translated_word, lang=target_lang).save(trans_audio_path)

    # Store latest
    st.session_state.latest = (word, translated_word, en_audio_path, trans_audio_path)
    st.session_state.history.append((word, translated_word))

# ------------------ Render Results ------------------
def render_results():
    col_latest, col_history = st.columns([1,1])
    
    # Latest Detection Card
    with col_latest:
        if st.session_state.latest:
            word, translated_word, en_audio, trans_audio = st.session_state.latest
            st.markdown("### ✨ Latest Detection")
            st.markdown(f"**English:** {word}")
            st.audio(en_audio)
            st.markdown(f"**{lang_choice}:** {translated_word}")
            st.audio(trans_audio)
    
    # History Card
    with col_history:
        st.markdown("### 📝 Detection History")
        if st.session_state.history:
            with st.expander("Show/Hide History"):
                for idx, (eng, trans) in enumerate(st.session_state.history,1):
                    st.markdown(f"{idx}. **English:** {eng} | **{lang_choice}:** {trans}")

# ------------------ Video Processing Function ------------------
def process_video_stream(cap):
    stframe = st.empty()
    last_word = None

    while cap.isOpened() and (st.session_state.run_webcam or st.session_state.run_video):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")

        if results[0].boxes:
            cls = int(results[0].boxes[0].cls[0])
            word = model.names[cls]
            if word != last_word:
                last_word = word
                show_prediction(word, target_lang, lang_choice)
                render_results()

    cap.release()
    cv2.destroyAllWindows()

# ------------------ Main ------------------
if mode == "Open Webcam":
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📸 Start Webcam"):
            st.session_state.run_webcam = True
            st.session_state.history = []
    with col2:
        if st.button("⏹️ Stop Webcam"):
            st.session_state.run_webcam = False

    if st.session_state.run_webcam:
        cap = cv2.VideoCapture(0)
        process_video_stream(cap)
    else:
        render_results()

elif mode == "Upload Video":
    uploaded_file = st.file_uploader("🎥 Upload a video file", type=["mp4","avi","mov"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start Video"):
                st.session_state.run_video = True
                st.session_state.history = []
        with col2:
            if st.button("⏹️ Stop Video"):
                st.session_state.run_video = False

        if st.session_state.run_video:
            cap = cv2.VideoCapture(video_path)
            process_video_stream(cap)
        else:
            render_results()
