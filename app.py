import os
import time
import uuid
import joblib
import cv2
import numpy as np
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage import filters
from skimage.util import view_as_windows
from scipy.ndimage import median_filter
from joblib import Parallel, delayed

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'static/uploads'
MODEL_FOLDER = 'models'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- CLASSES & COULEURS ---
CLASS_NAMES = ['Urban', 'Agriculture', 'Rangeland', 'Forest', 'Water', 'Barren']
# Couleurs assorties au CSS de l'interface
LABEL_TO_COLOR = {
    0: (74, 85, 104),    # Urban (Gris)
    1: (214, 158, 46),   # Agriculture (Jaune)
    2: (159, 122, 234),  # Rangeland (Violet)
    3: (56, 161, 105),   # Forest (Vert)
    4: (49, 130, 206),   # Water (Bleu)
    5: (229, 62, 62)     # Barren (Rouge)
}

# --- EXTRACTEUR DE FEATURES ---
class OptimizedFeatureExtractor:
    def __init__(self):
        self.distances = [1, 3]
        self.angles = [0, np.pi/4, np.pi/2]
    
    def extract(self, patch_rgb):
        patch = patch_rgb.astype(np.float32)
        gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
        features = []
        
        # 1. Stats Couleur
        for i in range(3):
            channel = patch[:, :, i]
            features.extend([
                channel.mean(), channel.std(),
                np.percentile(channel, 10), np.percentile(channel, 90)
            ])
        
        # 2. GLCM
        gray8 = (gray / 32).astype(np.uint8)
        glcm = graycomatrix(gray8, distances=self.distances, angles=self.angles,
                           levels=8, symmetric=True, normed=True)
        for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
            vals = graycoprops(glcm, prop).flatten()
            features.extend([vals.mean(), vals.std()])
        
        # 3. LBP
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
        features.extend(hist)
        
        # 4. Gradient
        sx = filters.sobel(gray, axis=0)
        sy = filters.sobel(gray, axis=1)
        grad_mag = np.sqrt(sx**2 + sy**2)
        features.extend([grad_mag.mean(), grad_mag.std()])
        
        return np.array(features, dtype=np.float32)

# --- CHARGEMENT DU MODÈLE ---
print("Chargement des modèles...")
try:
    model = joblib.load(os.path.join(MODEL_FOLDER, 'xgboost_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_FOLDER, 'scaler.pkl'))
    extractor = OptimizedFeatureExtractor()
    print("✅ Modèles chargés avec succès.")
except Exception as e:
    print(f"❌ ERREUR CRITIQUE : {e}")

# --- FONCTIONS UTILITAIRES ---

def array_to_base64(arr):
    """Convertit une image Numpy en chaîne Base64 pour le Web"""
    img = Image.fromarray(arr.astype('uint8'))
    buff = BytesIO()
    img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def process_image_fast(image_path):
    """Version vectorisée et optimisée"""
    # 1. Chargement
    img = np.array(Image.open(image_path).convert('RGB'))
    h, w = img.shape[:2]
    
    patch_size = 64
    stride = 32
    
    # 2. Découpage Intelligent (View as Windows)
    patches_window = view_as_windows(img, (patch_size, patch_size, 3), step=stride)
    n_rows, n_cols = patches_window.shape[:2]
    
    # Aplatir pour le ML
    flat_patches = patches_window.reshape(-1, patch_size, patch_size, 3)
    
    # 3. Extraction Parallèle (Utilisation de tous les coeurs)
    # batch_size='auto' améliore la distribution des tâches
    features_list = Parallel(n_jobs=-1, batch_size='auto')(
        delayed(extractor.extract)(patch) for patch in flat_patches
    )
    
    # 4. Prédiction en bloc
    X_new = np.nan_to_num(np.array(features_list))
    X_scaled = scaler.transform(X_new)
    preds = model.predict(X_scaled)
    
    # 5. Reconstruction VECTORISÉE (Instantannée)
    # On remet en grille
    preds_grid = preds.reshape(n_rows, n_cols)
    
    # On "étire" chaque prédiction pour remplir les pixels (remplace les boucles for)
    map_temp = preds_grid.repeat(stride, axis=0).repeat(stride, axis=1)
    
    # On place le résultat dans une image noire aux dimensions finales
    prediction_map = np.zeros((h, w), dtype=np.uint8)
    
    # Gestion des dimensions pour éviter les erreurs de bordures
    h_fill = min(h, map_temp.shape[0])
    w_fill = min(w, map_temp.shape[1])
    prediction_map[:h_fill, :w_fill] = map_temp[:h_fill, :w_fill]

    # 6. Lissage (taille 7 pour réduire l'effet bloc)
    clean_map = median_filter(prediction_map, size=7)
    
    # 7. Colorisation
    mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in LABEL_TO_COLOR.items():
        mask_vis[clean_map == label] = color
        
    return clean_map, mask_vis, img

# --- ROUTES FLASK ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'Aucune image'})
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Nom vide'})

        # Sauvegarde temporaire
        unique_filename = str(uuid.uuid4()) + '.png'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        # --- TRAITEMENT OPTIMISÉ ---
        start_time = time.time()
        pred_map, mask_vis, original_img = process_image_fast(file_path)
        proc_time = time.time() - start_time
        print(f"Image traitée en {proc_time:.2f} secondes")

        # --- STATISTIQUES (CORRECTION WATER) ---
        total_pixels = pred_map.size
        unique, counts = np.unique(pred_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        
        stats = []
        # On force la boucle sur TOUTES les classes (0 à 5)
        for i, name in enumerate(CLASS_NAMES):
            count = counts_dict.get(i, 0) # 0 si la classe est absente
            percent = round((count / total_pixels) * 100, 2)
            
            stats.append({
                'class_name': name,
                'percentage': percent
            })

        # Trier par pourcentage
        stats.sort(key=lambda x: x['percentage'], reverse=True)

        # Préparation retour JSON
        original_b64 = array_to_base64(original_img)
        result_b64 = array_to_base64(mask_vis)

        # Nettoyage
        try: os.remove(file_path)
        except: pass

        return jsonify({
            'success': True,
            'original_image': original_b64,
            'result_image': result_b64,
            'statistics': stats
        })

    except Exception as e:
        print(f"Erreur interne: {e}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)