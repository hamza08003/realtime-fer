from utils.face_detection import detect_and_crop_face, preprocess_image
from utils.results_manager import ResultsManager
from utils.inference import EmotionPredictor
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import zipfile
import shutil
import torch
import json
import cv2
import os



st.set_page_config(page_title="FER Web App", layout="wide", initial_sidebar_state="expanded")

with open("static/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


torch.classes.__path__ = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    predictor = EmotionPredictor("trained_model/AffectNet7_Model.pth", device)
except Exception as e:
    st.error(f"Error initializing the model: {str(e)}")
    st.error("Please ensure you have internet connection for first-time model download or check if the model file is corrupted.")
    st.stop()
    
results_manager = ResultsManager()


def get_emotion_class(emotion):
    return f"emotion-{emotion.lower()}"

# def display_confidence_scores(probabilities):
#     probabilities_dict = dict(zip(predictor.labels, probabilities))
#     sorted_list = sorted([(label, float(prob)) for label, prob in probabilities_dict.items()], key=lambda x: x[1], reverse=True)

#     st.markdown("### Confidence Scores")
#     for label, prob in sorted_list:
#         prob = min(100, max(0, prob)) 
#         st.write(f"{label}: {prob:.2f}%")
#         st.progress(prob / 100) 

def display_confidence_scores(probabilities):
    probs_dict = dict(zip(predictor.labels, probabilities))
    sorted_list = sorted([(label, float(prob)) for label, prob in probs_dict.items()], key=lambda x: x[1], reverse=True)

    html_content = ""
    for label, prob in sorted_list:
        label_id = label.lower().replace(" ", "-")
        prob_display = f"{prob:.2f}"
        html_content += f'''<div class="emotion-bar" id="{label_id}-bar">
                                <div class="label">{label}</div>
                                <div class="bar-container">
                                    <div class="bar" style="width: 0%;"></div>
                                </div>
                                <div class="percentage">{prob_display}%</div>
                            </div>'''

    probs_json = json.dumps(sorted_list).replace('"', '\\"')

    with open("static/confidence_scores.html", "r", encoding="utf-8") as file:
        html_template = file.read()

    html = html_template.replace("{{html_content}}", html_content).replace("{{probs_json}}", probs_json)
    components.html(html, height=400)


# ############################################################ #
#                        Sidebar Menu                          #
# ############################################################ #
st.markdown(
    """
    <style>
        /* Styles for the Options Menu container */
        [data-testid="stSidebarUserContent"] {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        
        iframe {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100% !important;
        }

        /* Styles for the st.image Container */
        [data-testid="stSidebar"] [data-testid="stImage"] {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 20px;
            margin-bottom: 10px; /* reduced margin to bring the menu closer to the logo */
            margin-top: -100px; /* move the logo up to center the group */
            margin-left: 50px; /* shift the logo slightly to the right */
        }

        /* Style for the image inside st.image */
        [data-testid="stSidebar"] [data-testid="stImage"] img  {
            width: 150px !important;
            height: 150px !important;
            object-fit: contain;
            border-radius: 50%;
            transition: transform 0.3s ease;
        }

        /* Hover effect for the image */
        div[data-testid="stImage"] img:hover {
            transform: scale(1.1);
        }

    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.image("static/facial-recognition.png", use_container_width=False)
    selected = option_menu(
        menu_title="Options",
        options=["About & Instructions", "Single Image(s)", "Multiple Images (ZIP)", "Camera Input", "Results"],
        icons=["info-circle", "image", "file-zip", "camera", "table"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {
                "padding": "5px"
            },
            "icon": {
                "color": "#00ffcc", 
                "font-size": "20px"
            },
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#00ccaa"
            },
            "nav-link-selected": {
                "color": "#00ffcc",
                "background-color": "#2c2c2c",
                "font-size": "16px"
            },
        },
    )


# ############################################################ #
#               About & Instructions Section                   #
# ############################################################ #
if selected == "About & Instructions":
    st.markdown(
    """
    <h1 style="text-align: center; color: #00ffcc; text-shadow: 0 0 15px rgba(0, 255, 204, 0.7);">
        🧠 Facial Emotion Recognition (FER) Tool
    </h1>
    <br><br>
    """,
    unsafe_allow_html=True
    )
    st.info(
        """
        ### ℹ️ About
        This web application uses a deep learning model (`ResEmoteNet`) to detect emotions from facial images or live webcam feed.
        It supports the following emotions:
        - 😐 **Neutral**
        - 😀 **Happiness**
        - 😢 **Sadness**
        - 😲 **Surprise**
        - 😨 **Fear**
        - 🤢 **Disgust**
        - 😡 **Anger**
        """
    )

    st.markdown("### 📌 Instructions")
    st.markdown(
        """
        1. **Single Image(s)**: Upload one or more images to detect emotions.
        2. **Multiple Images (ZIP)**: Upload a ZIP file containing images for batch processing.
        3. **Camera Input**: Use your webcam for real-time emotion detection.
        4. **Results**: View and download previous prediction results.
        """
    )

    st.write("")
    st.warning("**⚠ Note**: Ensure your images contain visible faces for accurate detection.")


# ############################################################ #
#                   Single Image(s) Section                    #
# ############################################################ #
elif selected == "Single Image(s)":
    st.markdown(
    """
    <h1 style="text-align: center; color: #00ffcc; text-shadow: 0 0 15px rgba(0, 255, 204, 0.7);">
        📷 FER On Single Image(s)
    </h1>
    <br><br>
    """,
    unsafe_allow_html=True
    )
    upd_files = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if upd_files:
        for upd_file in upd_files:
            st.subheader(f"Processing: {upd_file.name}")
            image = np.array(Image.open(upd_file))
            face_image = detect_and_crop_face(image)
            
            if face_image is None:
                st.error("No face detected in the image.")
                continue
            
            st.image(face_image, caption="Detected Face", use_container_width=True)
            input_tensor = preprocess_image(face_image).unsqueeze(0).to(device)
            pred_label, probs = predictor.predict(input_tensor)
            
            st.markdown(
                f"""
                <div class="prediction-box {get_emotion_class(pred_label)}">
                    Predicted Emotion: <strong>{pred_label}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )

            display_confidence_scores(probs)
            results_manager.add_result(upd_file.name, pred_label, probs)


# ############################################################ #
#                Multiple Images (ZIP) Section                 #
# ############################################################ #
elif selected == "Multiple Images (ZIP)":
    st.markdown(
    """
    <h1 style="text-align: center; color: #00ffcc; text-shadow: 0 0 15px rgba(0, 255, 204, 0.7);">
        🗁 FER On Multiple Images (ZIP)
    </h1>
    <br><br>
    """,
    unsafe_allow_html=True
    )

    uploaded_zip = st.file_uploader("Upload ZIP File", type=["zip"])
    
    if uploaded_zip:
        # Create a unique temporary directory
        temp_dir = tempfile.mkdtemp(prefix="fer_")
        
        try:
            with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            image_files = [f for f in os.listdir(temp_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if not image_files:
                st.error("No valid images found in the ZIP file.")
            else:
                for image_file in image_files:
                    st.subheader(f"Processing: {image_file}")
                    image_path = os.path.join(temp_dir, image_file)
                    image = cv2.imread(image_path)
                    face_image = detect_and_crop_face(image)
                    
                    if face_image is None:
                        st.error("No face detected in the image.")
                        continue
                    
                    st.image(face_image, caption="Detected Face", use_container_width=True)
                    input_tensor = preprocess_image(face_image).unsqueeze(0).to(device)
                    predicted_label, probabilities = predictor.predict(input_tensor)
                    
                    st.markdown(
                        f"""
                        <div class="prediction-box {get_emotion_class(predicted_label)}">
                            Predicted Emotion: <strong>{predicted_label}</strong>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Display confidence scores as animated bars
                    display_confidence_scores(probabilities)
                    
                    results_manager.add_result(image_file, predicted_label, probabilities)
        finally:
            # Use shutil.rmtree with error handling for more robust cleanup
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                st.warning(f"Note: Temporary files may need manual cleanup: {e}")


# ############################################################ #
#                    Camera Input Section                      #
# ############################################################ #
elif selected == "Camera Input":
    st.markdown(
    """
    <h1 style="text-align: center; color: #00ffcc; text-shadow: 0 0 15px rgba(0, 255, 204, 0.7);">
        📽️ Live Camera Emotion Detection
    </h1>
    <br><br>
    """,
    unsafe_allow_html=True
    )

    st.warning("Note: Camera input is only supported in a local Streamlit environment, not in hosted environments like Streamlit Cloud.")
    
    cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        st.error("Camera not available. Please ensure your webcam is connected and accessible.")
    else:
        run = st.checkbox("Start Camera")
        FRAME_WINDOW = st.image([])
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from camera.")
                break
            
            face_image = detect_and_crop_face(frame)
            if face_image:
                input_tensor = preprocess_image(face_image).unsqueeze(0).to(device)
                predicted_label, probabilities = predictor.predict(input_tensor)
                cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 
                st.write("Confidence Scores:")
                for label, prob in sorted(zip(predictor.labels, probabilities), key=lambda x: x[1], reverse=True):
                    st.write(f"{label}: {prob:.2f}%")
            
            FRAME_WINDOW.image(frame, channels="BGR")
        
        cap.release()


# ############################################################ #
#                       Results Section                        #
# ############################################################ #
elif selected == "Results":
    st.markdown(
    """
    <h1 style="text-align: center; color: #00ffcc; text-shadow: 0 0 15px rgba(0, 255, 204, 0.7);">
        📊 Prediction Results
    </h1>
    <br><br>
    """,
    unsafe_allow_html=True
    )
    results_df = results_manager.get_results()
    if results_df.empty:
        st.info("No results available yet. Run some predictions to see results here.")
    else:
        st.dataframe(results_df)
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="fer_results.csv",
            mime="text/csv"
        )
