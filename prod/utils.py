import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pandas as pd
from PIL import Image
from torchvision import transforms
from collections import defaultdict
import os

# ---------- Modelo con extracción de embedding ----------

class EmbeddingNet(nn.Module):
    def __init__(self, backbone='densenet121', embedding_size=128):
        super().__init__()
        self.embedding_size = embedding_size

        if backbone == 'densenet121':
            base_model = models.densenet121(weights=None)
            num_features = base_model.classifier.in_features
            base_model.classifier = nn.Identity()
            self.backbone = base_model
        else:
            raise ValueError("Backbone no soportado")

        self.embedding = nn.Linear(num_features, embedding_size)

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)
        return x

# ---------- Cargar modelo entrenado ----------

def load_trained_model(model_path, device='cpu', embedding_size=128):
    model = EmbeddingNet(backbone='densenet121', embedding_size=embedding_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ---------- Obtener embedding de una imagen ----------

def get_image_embedding(model, image_tensor, device='cpu'):
    model.eval()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        embedding = model(image_tensor.unsqueeze(0))
    return embedding.squeeze(0)

# ---------- Cargar imágenes desde CSV y generar embeddings por clase ----------

def compute_reference_embeddings(model, annotation_csv, transform, device='cpu', samples_per_class=5):
    df = pd.read_csv(annotation_csv)
    class_embeddings = defaultdict(list)

    for label in df['label'].unique():
        class_samples = df[df['label'] == label].sample(n=min(samples_per_class, len(df[df['label'] == label])), random_state=42)
        for _, row in class_samples.iterrows():
            path = row['path']
            if not os.path.exists(path):
                continue
            image = Image.open(path).convert('RGB')
            image = transform(image)
            embedding = get_image_embedding(model, image, device)
            class_embeddings[label].append(embedding)

    # Calcular promedio por clase
    reference_embeddings = {label: torch.stack(embeds).mean(0) for label, embeds in class_embeddings.items()}
    return reference_embeddings

# ---------- Predecir clase comparando con los embeddings de referencia ----------

def predict_character(model, image_tensor, reference_embeddings, device='cpu'):
    image_tensor = image_tensor.to(device)
    image_embedding = get_image_embedding(model, image_tensor, device)

    best_class = None
    best_similarity = float('-inf')

    for label, ref_embedding in reference_embeddings.items():
        similarity = F.cosine_similarity(image_embedding, ref_embedding, dim=0).item()
        if similarity > best_similarity:
            best_similarity = similarity
            best_class = label

    return best_class
