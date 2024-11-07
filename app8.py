import streamlit as st
import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Titre de l'application
st.title("Application de Prédiction de Segmentation d'Images")

# Uploader une image
uploaded_file = st.file_uploader("Choisissez une image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Afficher l'image téléchargée
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée', use_column_width=True)

    # Convertir l'image en bytes
    image_bytes = uploaded_file.getvalue()

    # Vérifiez le type de fichier et l'extension
    st.write(f"Type de fichier : {uploaded_file.type}")
    st.write(f"Nom du fichier : {uploaded_file.name}")

    # Envoyer l'image à l'API
    try:
        response = requests.post(
            "https://app8-c6adaba01656.herokuapp.com/predict",
            files={"image": (uploaded_file.name, image_bytes, uploaded_file.type)},
            verify=False  # Désactivez la vérification SSL (à éviter en production)
        )

        # Traiter la réponse
        if response.status_code == 200:
            # Afficher le masque prédit
            predicted_mask = response.json()['predicted_mask']
            
            # Convertir le masque en tableau NumPy pour l'affichage
            predicted_mask = np.array(predicted_mask)

            # Visualiser les résultats
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(image)
            ax[0].set_title("Image d'origine")
            ax[0].axis('off')
            
            ax[1].imshow(predicted_mask, cmap='jet', alpha=0.5)  # Afficher le masque
            ax[1].set_title("Masque prédit")
            ax[1].axis('off')

            st.pyplot(fig)
        else:
            st.error("Erreur lors de la prédiction : " + response.json().get('error', 'Erreur inconnue'))
    
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API : {e}")
