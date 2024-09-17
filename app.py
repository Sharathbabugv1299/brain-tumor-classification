import numpy as np
import os
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the saved model
model_test = load_model('fine_tuned_resnet50_brain_tumor.h5')

def predict_image(img_path, model):
    
    # Load the image and resize it to 224x224 (the size ResNet50 expects)
    img = image.load_img(img_path, target_size=(224, 224))
    
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    
    # Add an extra dimension (for batch size), since the model expects a batch of images
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess the image (this is necessary because ResNet50 was trained with certain preprocessing)
    img_array = preprocess_input(img_array)
    
    # Predict the class (probability) of the image
    prediction = model_test.predict(img_array)
    
    # Since it's a binary classification (0 or 1), round the prediction to get the class
    predicted_class = np.round(prediction).astype(int)
    
    # Output the prediction
    if predicted_class == 0:
        #print("The model predicts: Brain Tumor")
        return "Brain Tumor"
    else:
        #print("The model predicts: Healthy")
        return "Healthy"

# Initialize Flask app
app = Flask(__name__)

# Ensure the 'uploads' folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define the main route to handle image upload
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if an image was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded.'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected.'})

        # Save the uploaded image to a temporary path
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)

        # Make prediction on the uploaded image
        predicted_label = predict_image(img_path, model_test)

        # Clean up: remove the image after prediction
        os.remove(img_path)

        # Return the prediction as JSON
        return jsonify({'prediction': f'The model predicts: {predicted_label}'})
    
    return render_template('upload.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

