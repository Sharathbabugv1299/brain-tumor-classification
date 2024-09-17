import os
from flask import Flask, request, render_template, jsonify
from gradio_client import Client, handle_file

# Initialize Gradio Client
client = Client("Hareeharan03/Brain-Tumor_classification")

# Define the Gradio API endpoint name
API_NAME = "/predict"

def predict_image(img_path):
    # Use gradio_client to send the image file to the API
    result = client.predict(
        img=handle_file(img_path),  # Pass the image file path to the API
        api_name=API_NAME
    )
    return result

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

        # Make prediction on the uploaded image using the Gradio API
        try:
            prediction = predict_image(img_path)
        except Exception as e:
            # Handle exceptions if the API call fails
            return jsonify({'error': str(e)})

        # Clean up: remove the image after prediction
        os.remove(img_path)

        # Return the prediction as JSON
        return jsonify({'prediction': f'The model predicts: {prediction}'})
    
    return render_template('upload.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
