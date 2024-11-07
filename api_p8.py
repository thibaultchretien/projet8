from flask import Flask, request, jsonify
from flask_cors import CORS  # Pour gérer les CORS
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)  # Activer CORS

# Chargez votre modèle pré-entraîné
model = load_model('segmentation_model.h5')

@app.route('/')
def home():
    return "Test de l'API local pour le projet 8"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image fournie.'}), 400
    
    file = request.files['image']
    
    # Vérifier le type de fichier
    if not file.filename.endswith(('png', 'jpg', 'jpeg')):
        return jsonify({'error': 'Type de fichier non valide. Seules les images PNG et JPEG sont acceptées.'}), 400
    
    try:
        # Charger l'image
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((256, 256))  # Ajustez la taille selon votre modèle
        img_array = np.array(img) / 255.0  # Normaliser l'image
        img_array = np.expand_dims(img_array, axis=0)  # Ajouter la dimension de lot

        # Faire la prédiction
        pred_mask = model.predict(img_array)
        pred_mask = np.argmax(pred_mask, axis=-1).reshape(256, 256)  # Ajuster la forme

        # Retourner le masque prédit sous forme de liste
        response = {
            'predicted_mask': pred_mask.tolist()  # Convertir en liste pour le JSON
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': 'Erreur lors de la prédiction.'}), 500

if __name__ == '__main__':
    # Utiliser le port donné par Heroku
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
