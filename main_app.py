# Library imports
import os
# Reduce TensorFlow/absl verbosity before i mporting TF
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # 0=all,1=info,2=warning,3=error
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')  # disable oneDNN custom ops info message

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import logging
logging.getLogger('absl').setLevel(logging.ERROR)

import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf

# Get the absolute path to the model
model_path = os.path.join(os.path.dirname(__file__), 'plant_disease_model.h5')

# Loading the Model (compile=False avoids optimizer/metrics restore warnings for inference)
try:
    model = load_model(model_path, compile=False)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Try to open the app in the default browser (safe, non-fatal)
# This avoids relying on PowerShell Start-Process and prevents the "file not found" error.
try:
    import webbrowser, threading, time
    def _open_browser_once():
        # Delay briefly to allow Streamlit server to start
        time.sleep(1)
        url = "http://localhost:8501"
        try:
            webbrowser.open_new_tab(url)
        except Exception as e:
            # Non-fatal: print an informational message to the terminal
            print(f"Auto-open browser failed: {e}")
    threading.Thread(target=_open_browser_once, daemon=True).start()
except Exception as e:
    print(f"Browser opener init failed: {e}")
                    
# Logging setup for terminal output
import logging
logger = logging.getLogger('plant_app')
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)

print("Starting Plant Disease Detection Streamlit app...")
logger.info("Streamlit app starting")

# Name of Classes
CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Early_blight', 'Corn-Common_rust')

# Setting Title of App
st.title("Plant Disease Detection")
st.markdown("Upload an image of the plant leaf")

# Uploading the plant image
plant_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict Disease')

# On predict button click
if submit:
    if plant_image is not None:
        try:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            
            if opencv_image is None:
                st.error("Error: Could not read the image. Please upload a valid JPG image.")
            else:
                # Displaying the image
                st.image(opencv_image, channels="BGR", caption="Uploaded Plant Leaf")
                st.write(f"Image Shape: {opencv_image.shape}")
                
                # Resizing the image
                resized_image = cv2.resize(opencv_image, (256, 256))
                
                # Normalize the image
                resized_image = resized_image.astype('float32') / 255.0
                input_image = np.reshape(resized_image, (1, 256, 256, 3))
                
                # Make Prediction
                Y_pred = model.predict(input_image, verbose=0)
                confidence = np.max(Y_pred) * 100
                result = CLASS_NAMES[np.argmax(Y_pred)]
                
                plant_name = result.split('-')[0]
                disease_name = result.split('-')[1]

                # Log prediction to terminal (safe fallback to print)
                try:
                    logger.info(f"Prediction -> Plant: {plant_name}, Disease: {disease_name}, Confidence: {confidence:.2f}%")
                except Exception:
                    print(f"Prediction -> {plant_name} | {disease_name} | {confidence:.2f}%")

                # Display Results
                st.success("Prediction Complete!")
                st.markdown("---")
                
                st.markdown(f"### 🌿 Plant: **{plant_name}**")
                st.markdown(f"### 🐛 Disease: **{disease_name}**")
                st.markdown(f"### 📊 Confidence: **{confidence:.2f}%**")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please upload an image first!")
