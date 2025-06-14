import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Resize function
def preprocess_image(img):
    img = img.resize((224, 224))  # Adjust to your Teachable Machine input size
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction function
def predict(img):
    img_array = preprocess_image(img)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output)
    confidence = np.max(output)
    return prediction, confidence

# Label map (adjust to your own labels)
labels = ["Normal", "Crack", "Severe Crack", "Minor Crack"]

# Streamlit UI
st.title("Anomaly Detection App (Upload or Live Camera)")

option = st.radio("Choose input method:", ("Upload Image", "Use Camera"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        pred, conf = predict(img)
        st.success(f"Prediction: {labels[pred]} (Confidence: {conf:.2f})")

elif option == "Use Camera":

    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_resized = cv2.resize(img, (224, 224))
            img_array = img_resized.astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            pred = np.argmax(output)
            label = labels[pred]

            # Draw label
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return img

    webrtc_streamer(key="camera", video_transformer_factory=VideoTransformer)
