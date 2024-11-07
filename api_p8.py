import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64

app = Flask(__name__)

# Charger le modèle
model = load_model('model_unet.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer l'image envoyée en base64
        data = request.get_json()
        img_data = data.get('image')

        if not img_data:
            return jsonify({'error': 'No image provided'}), 400

        # Décoder l'image de base64
        img_data = img_data.split(',')[1]
        img_bytes = base64.b64decode(img_data)

        # Ouvrir l'image et la convertir en format compatible avec le modèle
        img = Image.open(io.BytesIO(img_bytes))
        img = img.resize((256, 256))  # Redimensionner l'image à la taille d'entrée du modèle
        img = np.array(img) / 255.0  # Normalisation des pixels
        img = np.expand_dims(img, axis=0)  # Ajouter la dimension du batch

        # Prédiction
        prediction = model.predict(img)
        mask = prediction[0]  # Supposer que la sortie est un masque de même taille que l'image

        # Convertir le masque en image pour l'envoyer au client
        mask = (mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask)

        # Convertir le masque en base64 pour le renvoyer au client
        buffered = io.BytesIO()
        mask_img.save(buffered, format="PNG")
        mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({'mask': mask_base64})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)
