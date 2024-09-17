import numpy as np
import os
import requests
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Function to download the model file from GitHub
def download_model(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Model downloaded and saved as {filename}.")
    except Exception as e:
        print(f"Error downloading the model: {e}")

# URL of the raw model file on GitHub
model_url = 'https://github.com/Sharathbabugv1299/brain-tumor-classification-Using_DeepLearning/raw/main/fine_tuned_resnet50_brain_tumor.h5'
model_path = 'fine_tuned_resnet50_brain_tumor.h5'

# Download the model if it doesn't exist
if not os.path.exists(model_path):
    download_model(model_url, model_path)

def load_model_check():
    if not os.path.exists(model_path):
        download_model(model_url, model_path)
        if not os.path.exists(model_path):
            print("Model download failed. Exiting.")
            return None
    
    try:
        print("Loading model...")
        model_test = load_model(model_path)
        print("Model loaded successfully.")
        return model_test
    except OSError as e:
        print(f"OSError while loading the model: {e}")
        return None
    except Exception as e:
        print(f"General error while loading the model: {e}")
        return None

# Load the model
model_test = load_model_check()

def predict_image(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    prediction = model.predict(img_array)
    predicted_class = np.round(prediction).astype(int)
    return "Brain Tumor" if predicted_class == 0 else "Healthy"

# Initialize Flask app
app = Flask(__name__)

# Ensure the 'uploads' folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded.'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected.'})

        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)

        predicted_label = predict_image(img_path, model_test)
        os.remove(img_path)
        return jsonify({'prediction': f'The model predicts: {predicted_label}'})
    
    return render_template('upload.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
