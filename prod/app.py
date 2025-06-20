import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from utils import load_trained_model, predict_character

# Configuración general
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'modelo.pth'
EMBEDDINGS_PATH = 'data/reference_embeddings.pt'  # archivo generado previamente

# Cargar modelo entrenado
st.info("Cargando modelo...")
model = load_trained_model(MODEL_PATH, device=DEVICE)

# Cargar embeddings de referencia
st.info("Cargando embeddings de referencia...")
reference_embeddings = torch.load(EMBEDDINGS_PATH, map_location=DEVICE)

# Preprocesamiento de imágenes (debe coincidir con entrenamiento)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Interfaz
st.title("🔍 Clasificador de Personajes de Los Simpsons (vía embeddings)")

uploaded_file = st.file_uploader("Subí una imagen de un personaje", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Preprocesar imagen
    image_tensor = transform(image)

    # Predecir personaje
    prediction = predict_character(model, image_tensor, reference_embeddings, device=DEVICE)

    st.success(f"🎯 Personaje predicho: **{prediction}**")
