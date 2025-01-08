import streamlit as st
import cv2
import pyttsx3
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load the YOLO model
model = YOLO(r"D:\ASL Sign to text\best.pt")  # Replace with the path to your model

def process_frame(frame):
    """Process the frame for YOLO detection."""
    results = model.predict(source=frame, conf=0.6, save=False, save_txt=False)
    detections = results[0].boxes.data  # Get bounding boxes and scores

    detected_classes = []
    if detections is not None and len(detections) > 0:
        # Get the first detection only
        x1, y1, x2, y2, conf, cls = detections[0][:6]
        cls = int(cls)
        class_name = results[0].names[cls]
        detected_classes.append(class_name)

        # Draw the bounding box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            class_name,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return frame, detected_classes

def speak_detected_objects(detected_classes):
    """Convert detected objects to speech."""
    if detected_classes:
        # Initialize a new pyttsx3 engine for each TTS event
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 130)
        tts_engine.setProperty('volume', 0.9)
        detected_sentence = ", ".join(detected_classes)
        tts_engine.say(detected_sentence)
        tts_engine.runAndWait()
        tts_engine.stop()

# CSS styling
st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f4;
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background-image: linear-gradient(to bottom right, #ffffff, #e6e6e6);
        background-attachment: fixed;
    }
    .header {
        font-size: 40px;
        color: #4CAF50;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .info {
        font-size: 18px;
        color: #555555;
        text-align: center;
        margin-top: 10px;
    }
    .labels {
        font-size: 24px;
        color: #333333;
        margin-top: 20px;
        text-align: center;
        font-weight: bold;
    }
    .creative-text {
        font-size: 30px;
        color: #4CAF50;
        text-align: center;
        font-style: italic;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def main():
    st.markdown('<div class="header">SIGNIFY</div>', unsafe_allow_html=True)
    st.sidebar.header("Controls")

    # Create buttons for start, stop detection, and toggle mic
    start_detection = st.sidebar.button("Start Detection")
    stop_detection = st.sidebar.button("Stop Detection")
    toggle_mic = st.sidebar.checkbox("Enable Microphone for TTS", value=False)

    frame_placeholder = st.empty()
    label_placeholder = st.empty()
    creative_placeholder = st.empty()

    if start_detection:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not access the webcam.")
            return

        st.markdown('<div class="info">Detection Started. Press Stop Detection to end.</div>', unsafe_allow_html=True)

        while not stop_detection:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Failed to capture frame from webcam.")
                break

            # Process frame for detections
            frame, detected_classes = process_frame(frame)

            # Update detected labels
            if detected_classes:
                creative_placeholder.markdown(
                    f'<div class="creative-text">Detected Sign: {" ".join(detected_classes)}</div>',
                    unsafe_allow_html=True,
                )
                label_placeholder.markdown(f'<div class="labels">Detected: {", ".join(detected_classes)}</div>', unsafe_allow_html=True)
                if toggle_mic:
                    speak_detected_objects(detected_classes)  # Speak detected objects
            else:
                creative_placeholder.markdown('<div class="creative-text">No Sign Detected</div>', unsafe_allow_html=True)
                label_placeholder.markdown('<div class="labels">No objects detected.</div>', unsafe_allow_html=True)

            # Convert BGR to RGB for Streamlit
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB")

        cap.release()
        cv2.destroyAllWindows()
        st.markdown('<div class="info">Detection Stopped.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
