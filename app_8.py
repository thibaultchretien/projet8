import os
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

# Load your pre-trained model
model = load_model('segmentation_model.h5')  

@app.route('/')
def home():
    return "test de l api local pour le projet 8 "

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image fournie.'}), 400

    file = request.files['image']

    # Check file type
    if not file.filename.endswith(('png', 'jpg', 'jpeg')):
        return jsonify({'error': 'Type de fichier non valide. Seules les images PNG et JPEG sont acceptées.'}), 400

    try:
        # Load the image
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((256, 256))  # Adjust size according to your model
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make the prediction
        pred_mask = model.predict(img_array)
        pred_mask = np.argmax(pred_mask, axis=-1).reshape(256, 256)  # Adjust shape

        # Return the predicted mask as a list
        response = {
            'predicted_mask': pred_mask.tolist()  # Convert to list for JSON
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': 'Erreur lors de la prédiction.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))  # Default to 5001 for local development
    app.run(host='0.0.0.0', port=port, debug=True)  # Bind to the assigned port
