import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_erosion, binary_dilation, gaussian_filter, convolve
from scipy.signal import convolve2d
from skimage import measure, color, exposure
from skimage.draw import line, disk
from skimage.transform import resize
import re
from scipy.stats import norm
from scipy import signal

###############################################################################
#                          PARTIE 1 :
###############################################################################

barcode_height, barcode_width = 109, 249

def analyze_image_properties(image):
    """Analyse les propriétés de l'image (taille, bruit, contraste)."""
    height, width = image.shape
    noise_level = np.std(image - gaussian_filter(image, sigma=1.0))
    p5, p95 = np.percentile(image, (5, 95))
    contrast_level = p95 - p5
    return {
        'image_size': (height, width),
        'noise_level': noise_level,
        'contrast_level': contrast_level
    }

def suggest_parameters(image, barcode_height, barcode_width):
    """Suggère des paramètres de traitement en fonction de la hauteur/largeur attendues et des propriétés de l'image."""
    properties = analyze_image_properties(image)
    sigma_deriv = max(1.0, barcode_width / 95 / 2)
    sigma_tensor = max(2.0, barcode_width / 20)
    window_size = 2 * int(3 * sigma_tensor) + 1
    window_size = min(window_size, barcode_height // 2)

    if properties['noise_level'] > 0.1:
        sigma_deriv *= 1.5
        sigma_tensor *= 1.5

    min_area = barcode_width * barcode_height * 0.3
    if properties['contrast_level'] < 0.3:
        min_area *= 0.8

    return {
        'sigma_deriv': sigma_deriv,
        'sigma_tensor': sigma_tensor,
        'window_size': window_size,
        'region_params': {
            'min_area': min_area,
            'min_ratio': 1.5,
            'max_ratio': 8.0
        }
    }

def preprocess_image(image):
    """
    Améliore le contraste de l'image en appliquant un filtrage gaussien,
    une égalisation adaptative, etc.
    """
    image_float = image.astype(float)
    image_norm = image_float

    noise_level = 0.05
    noisy_image = image_norm + np.random.normal(0, noise_level, image_norm.shape)
    noisy_image = np.clip(noisy_image, 0, 1)

    image_smooth = gaussian_filter(noisy_image, sigma=1.0)
    image_eq = exposure.equalize_adapthist(image_smooth, clip_limit=0.03)

    p2, p98 = np.percentile(image_eq, (2, 98))
    image_contrast = exposure.rescale_intensity(image_eq, in_range=(p2, p98))
    return image_contrast

def preprocess_imagee(image):
    """Convertit l'image en niveaux de gris si nécessaire (et effectue une binarisation simple si souhaité)."""
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
    return image

def load_image(image_path):
    """Charge et redimensionne l'image en niveaux de gris."""
    image = plt.imread(image_path)
    if image.ndim == 3:
        if image.shape[-1] == 4:
            image = image[..., :3]
        image = color.rgb2gray(image)
    image = resize_image(image)
    return image

def filtre_gaussien_RIF(sigma, direction='x'):
    """Crée un filtre gaussien RIF pour le calcul des gradients."""
    size = 2 * int(3 * sigma) + 1
    x = np.linspace(-3*sigma, 3*sigma, size)
    y = np.linspace(-3*sigma, 3*sigma, size)
    X, Y = np.meshgrid(x, y)
    
    terme_commun = np.exp(-(X**2 + Y**2)/(2*sigma**2))
    if direction == 'x':
        noyau = -X/(np.pi*sigma**4) * terme_commun
    else:
        noyau = -Y/(np.pi*sigma**4) * terme_commun

    noyau_pos = noyau > 0
    noyau_neg = noyau < 0
    if np.any(noyau_pos):
        noyau[noyau_pos] /= np.sum(noyau[noyau_pos])
    if np.any(noyau_neg):
        noyau[noyau_neg] /= -np.sum(noyau[noyau_neg])

    return noyau

def analyser_regions(mask):
    """Analyse les régions détectées dans un masque binaire (aire, ratio d'aspect, etc.)."""
    labels = measure.label(mask)
    regions = measure.regionprops(labels)
    
    result = []
    for region in regions:
        y0, x0, y1, x1 = region.bbox
        info = {
            'label': region.label,
            'aire': region.area,
            'ratio_aspect': (x1 - x0) / (y1 - y0) if (y1 - y0) > 0 else 0,
            'bbox': (x0, y0, x1, y1),
            'orientation': region.orientation,
            'excentricite': region.eccentricity
        }
        result.append(info)
    
    return result

def filtrer_regions(regions, min_area=1000, min_ratio=1.5, max_ratio=8.0):
    """Filtre les régions selon divers critères (aire, ratio d'aspect, etc.)."""
    regions_filtrees = []
    for region in regions:
        if (region['aire'] > min_area and 
            min_ratio < region['ratio_aspect'] < max_ratio):
            regions_filtrees.append(region)
    
    if len(regions_filtrees) > 1:
        regions_filtrees.sort(key=lambda r: r['aire'], reverse=True)
        regions_filtrees = regions_filtrees[:1]
    
    return regions_filtrees

def garder_plus_grande_region(mask):
    """Ne conserve que la plus grande région connectée du masque."""
    labels = measure.label(mask)
    if labels.max() == 0:
        return np.zeros_like(mask)
    regions = measure.regionprops(labels)
    aires = [region.area for region in regions]
    index_max = np.argmax(aires)
    nouveau_mask = (labels == regions[index_max].label)
    
    return nouveau_mask

def dessiner_axe_principal(image, mask):
    """
    Calcule l'axe principal d'une région dans le masque,
    dessine cet axe sur l'image et calcule quelques points aux extrémités.
    """
    y, x = np.where(mask)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_centered = x - x_mean
    y_centered = y - y_mean
    cov_matrix = np.zeros((2, 2))
    cov_matrix[0, 0] = np.mean(x_centered**2)
    cov_matrix[1, 1] = np.mean(y_centered**2)
    cov_matrix[0, 1] = np.mean(x_centered * y_centered)
    cov_matrix[1, 0] = cov_matrix[0, 1]
    
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    principal_val = np.max(eigenvals)
    principal_vec = eigenvecs[:, np.argmax(eigenvals)]
    
    projections = x_centered * principal_vec[0] + y_centered * principal_vec[1]
    max_proj = np.max(projections)
    min_proj = np.min(projections)
    extension_factor = 1.15

    point1 = np.array([
        x_mean + extension_factor * min_proj * principal_vec[0],
        y_mean + extension_factor * min_proj * principal_vec[1]
    ])
    point2 = np.array([
        x_mean + extension_factor * max_proj * principal_vec[0],
        y_mean + extension_factor * max_proj * principal_vec[1]
    ])

    result = np.stack([image]*3, axis=-1) if image.ndim == 2 else image.copy()
    
    std = np.sqrt(principal_val) / 4
    n_points = 4
    points_list1 = []
    points_list2 = []
    
    for i, point in enumerate([point1, point2]):
        noise_x = np.random.normal(0, std, n_points)
        noise_y = np.random.normal(0, std, n_points)
        points = point + np.column_stack((noise_x, noise_y))
        
        for px, py in points:
            px_int, py_int = int(px), int(py)
            if 0 <= py_int < image.shape[0] and 0 <= px_int < image.shape[1]:
                if i == 0:
                    points_list1.append([px_int, py_int])
                else:
                    points_list2.append([px_int, py_int])
                result[py_int, px_int] = [1, 0, 0]
    
    theta = np.arctan2(principal_vec[1], principal_vec[0])
    return result, theta, point1, point2, points_list1, points_list2

def extraire_region_code_barres(image, regions, point1, point2):
    """
    Extrait la région la plus probable du code-barres dans l'image (si la région filtrée existe),
    la binarise et calcule des points approximatifs de scan (point1_cropped, point2_cropped).
    """
    if not regions:
        return None, None, None, None
    
    region = regions[0]
    x0, y0, x1, y1 = region['bbox']
    height, width = image.shape[:2]
    
    # Calcul des marges
    margin_x = int((x1 - x0) * 0.1)
    margin_y = int((y1 - y0) * 0.1)
    
    crop_x0 = max(0, x0 - margin_x)
    crop_y0 = max(0, y0 - margin_y)
    crop_x1 = min(width, x1 + margin_x)
    crop_y1 = min(height, y1 + margin_y)
    
    cropped_image = image[crop_y0:crop_y1, crop_x0:crop_x1]
    binary_cropped = preprocess_imagee(cropped_image)
    
    cropped_height, cropped_width = binary_cropped.shape[:2]
    
    # Positionnement d'un segment horizontal au milieu de l'image rognée
    point1_cropped = [0, cropped_height // 2]
    point2_cropped = [cropped_width, cropped_height // 2]
    
    return cropped_image, binary_cropped, point1_cropped, point2_cropped

def calculer_gradients(image, sigma_deriv):
    """Calcule les gradients normalisés (Ix, Iy)."""
    Ix = convolve2d(image, filtre_gaussien_RIF(sigma_deriv, 'x'), mode='same', boundary='symm')
    Iy = convolve2d(image, filtre_gaussien_RIF(sigma_deriv, 'y'), mode='same', boundary='symm')
    
    norme = np.sqrt(Ix**2 + Iy**2)
    mask = norme > 0
    
    Ix_norm = np.zeros_like(Ix)
    Iy_norm = np.zeros_like(Iy)
    
    Ix_norm[mask] = Ix[mask] / norme[mask]
    Iy_norm[mask] = Iy[mask] / norme[mask]
    
    return Ix_norm, Iy_norm

def calculer_tenseur(Ix, Iy, sigma_tensor, window_size):
    """Calcule le tenseur de structure (Txx, Txy, Tyy)."""
    x = np.linspace(-(window_size//2), window_size//2, window_size)
    y = np.linspace(-(window_size//2), window_size//2, window_size)
    X, Y = np.meshgrid(x, y)
    W = np.exp(-(X**2 + Y**2)/(2*sigma_tensor**2))
    W = W / W.sum()
    
    Txx = convolve2d(Ix * Ix, W, mode='same', boundary='symm')
    Txy = convolve2d(Ix * Iy, W, mode='same', boundary='symm')
    Tyy = convolve2d(Iy * Iy, W, mode='same', boundary='symm')
    
    return Txx, Txy, Tyy

def calculer_coherence(Txx, Txy, Tyy):
    """Calcule la mesure de cohérence (D) à partir du tenseur de structure."""
    numerateur = np.sqrt((Txx - Tyy)**2 + 4*Txy**2)
    denominateur = Txx + Tyy
    
    mask = denominateur > 0
    D = np.zeros_like(Txx)
    D[mask] = numerateur[mask] / denominateur[mask]
    
    return D

def operations_morphologiques(mask):
    """Applique des opérations morphologiques (dilation/érosion) pour nettoyer le masque."""
    element_large = np.ones((7, 7), dtype=bool)
    element_small = np.ones((3, 3), dtype=bool)
    
    mask = binary_dilation(mask, structure=element_large, iterations=4)
    mask = binary_erosion(mask, structure=element_large, iterations=4)
    
    mask = binary_erosion(mask, structure=element_small)
    mask = binary_dilation(mask, structure=element_small)
    
    return mask

def resize_image(image, max_width=900, max_height=600):
    """
    Redimensionne l'image pour ne pas dépasser (max_width, max_height).
    Ne l'agrandit pas si elle est déjà plus petite.
    """
    height, width = image.shape[:2]
    
    width_ratio = max_width / width
    height_ratio = max_height / height
    scale = min(width_ratio, height_ratio)
    
    if scale >= 1:
        return image

    new_width = int(width * scale)
    new_height = int(height * scale)
    
    resized_image = resize(image, (new_height, new_width), anti_aliasing=True, preserve_range=True)
    return resized_image

def process_image_generate(image_path):
    """
    Traite l'image pour détecter la région où se trouve potentiellement le code-barres
    et crée deux sorties :
      1) L'image rognée
      2) L'image binarisée
    avec des points de scan inclus dans le nom de fichier.
    """
    print("  [Génération] Chargement et prétraitement de l'image...")
    image = load_image(image_path)
    params = suggest_parameters(image, barcode_height, barcode_width)
    
    # Calcul des gradients et du tenseur
    Ix, Iy = calculer_gradients(image, params['sigma_deriv'])
    Txx, Txy, Tyy = calculer_tenseur(Ix, Iy, params['sigma_tensor'], params['window_size'])
    D = calculer_coherence(Txx, Txy, Tyy)
    
    seuil = max(0.7, np.percentile(D, 95) * 0.8)
    mask = D > seuil
    
    # Opérations morphologiques et extraction de la plus grande région
    mask = operations_morphologiques(mask)
    mask = garder_plus_grande_region(mask)
    
    # Détermination des régions et de l'axe principal
    regions = analyser_regions(mask)
    regions_filtrees = filtrer_regions(regions, **params['region_params'])
    image_avec_axe, theta, point1, point2, points_list1, points_list2 = dessiner_axe_principal(image, mask)
    
    # Extraction de la région du code-barres
    region_barcode, binary_barcode, point1_cropped, point2_cropped = extraire_region_code_barres(
        image_avec_axe, regions_filtrees, point1, point2
    )
    
    return region_barcode, binary_barcode, point1_cropped, point2_cropped

def process_directory_generate(input_dir, output_dir):
    """
    Parcourt toutes les images dans input_dir et génère des images
    “processed_*” dans output_dir pour la phase de décodage.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    total_processed = 0
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(image_extensions):
            input_path = os.path.join(input_dir, filename)
            print(f"\n[Génération] Traitement de {filename}...")
            
            try:
                region_barcode, binary_barcode, point1_cropped, point2_cropped = process_image_generate(input_path)
                
                if region_barcode is not None and point1_cropped is not None:
                    base_name = os.path.splitext(filename)[0]
                    p1_str = f"p1_{int(point1_cropped[0])}_{int(point1_cropped[1])}"
                    p2_str = f"p2_{int(point2_cropped[0])}_{int(point2_cropped[1])}"
                    
                    # Sauvegarde de l'image rognée
                    output_filename = f"processed_{base_name}_{p1_str}_{p2_str}.png"
                    output_path = os.path.join(output_dir, output_filename)
                    plt.imsave(output_path, region_barcode, cmap='gray')
                    
                    # Sauvegarde de l'image binarisée
                    binary_filename = f"processed_{base_name}_{p1_str}_{p2_str}_bin.png"
                    binary_path = os.path.join(output_dir, binary_filename)
                    plt.imsave(binary_path, binary_barcode, cmap='gray')
                    
                    print(f"  ✓ Images sauvegardées dans {output_dir}:")
                    print(f"    - {output_filename}")
                    print(f"    - {binary_filename}")
                    print(f"    (Points: {p1_str}, {p2_str})")
                    total_processed += 1
                else:
                    print(f"  ✗ Aucune région code-barres détectée dans {filename}")
            except Exception as e:
                print(f"  ✗ Erreur lors du traitement de {filename}: {str(e)}")
                continue

    print(f"\n[Génération] Traitement terminé !")
    print(f"Nombre d'images traitées avec succès: {total_processed}")

def main_generate():
    """
    Fonction 'main' pour la génération des images processed_ (issue de code2_B).
    """
    input_dir = "Base_donnes"
    output_dir = "OUTPUT_BARCODE"

    if not os.path.exists(input_dir):
        print(f"Erreur : Le dossier d'entrée '{input_dir}' n'existe pas.")
        return
    
    process_directory_generate(input_dir, output_dir)

###############################################################################
#         PARTIE 2 :
###############################################################################

def extract_points_from_filename(filename):
    """Extrait les points de scan du nom de fichier."""
    try:
        pattern = r'p1_(\d+)_(\d+)_p2_(\d+)_(\d+)'
        match = re.search(pattern, filename)
        if match:
            x1, y1, x2, y2 = map(int, match.groups())
            return (x1, y1), (x2, y2)
    except Exception as e:
        print(f"Erreur lors de l'extraction des points: {e}")
    return None, None

def extract_barcode_from_filename(filename):
    """Extrait le code-barres attendu du nom de fichier s'il est sous la forme : 'processed_<CODE>-..."""
    try:
        barcode = filename.split('processed_')[1].split('-')[0]
        if barcode.isdigit():
            return barcode
    except:
        pass
    return None

def extract_signature(img_array, x0, y0, x1, y1, num_samples):
    """
    Extrait la signature (vecteur d'intensités) le long du segment défini par (x0,y0) et (x1,y1)
    en interpolant sur num_samples points.
    Retourne : signature, x_coords, y_coords.
    """
    x_coords = np.linspace(x0, x1, num_samples).astype(int)
    y_coords = np.linspace(y0, y1, num_samples).astype(int)
    signature = img_array[y_coords, x_coords]
    return signature, x_coords, y_coords

def otsu_threshold(hist):
    """
    Calcule le seuil d'Otsu à partir d'un histogramme (simple implémentation).
    Ici, on retourne l'indice du pic maximal.
    """
    threshold = np.argmax(hist)
    return threshold

def find_limits(binary_signature):
    """
    Trouve les indices gauche et droit de la partie utile de la signature binaire.
    Retourne le premier et le dernier indice où la valeur est 1.
    """
    indices = np.where(binary_signature == 1)[0]
    if indices.size == 0:
        return None, None
    left_limit = indices[0]
    right_limit = indices[-1]
    return left_limit, right_limit

def calculate_optimal_samples(signature_length):
    """
    Calcule un nouveau nombre d'échantillons et un coefficient u (échantillons par unité de base).
    Pour cet exemple, u est le ratio arrondi de signature_length / 95.
    """
    u = max(1, int(round(signature_length / 95)))
    num_samples_new = 95 * u
    return num_samples_new, u

def segment_barcode(binary_signature, u):
    """
    Segmente la signature binaire en 12 parties (6 pour la première moitié et 6 pour la seconde)
    correspondant aux 12 chiffres codés (le premier chiffre étant implicite).
    Retourne un dictionnaire avec deux listes de segments.
    """
    total_samples = len(binary_signature)
    segment_length = total_samples // 12
    segments = {
        'first_group': [binary_signature[i*segment_length:(i+1)*segment_length] for i in range(6)],
        'second_group': [binary_signature[(6+i)*segment_length:(7+i)*segment_length] for i in range(6)]
    }
    return segments

def decode_first_group(segments, u):
    """
    Décode le premier groupe de segments et retourne le premier chiffre (implicite)
    ainsi qu'une liste des 6 chiffres décodés.
    Ici, la fonction retourne des valeurs fictives pour l'exemple.
    """
    first_digit = "0"
    first_group_digits = ["1", "2", "3", "4", "5", "6"]
    return first_digit, first_group_digits

def decode_second_group(segments, u):
    """
    Décode le second groupe de segments et retourne une liste des 6 chiffres décodés.
    Ici, la fonction retourne des valeurs fictives pour l'exemple.
    """
    second_group_digits = ["7", "8", "9", "0", "1", "2"]
    return second_group_digits

def process_single_image_decode(image_path, output_dir):
    """
    Traite une image (prétraitée lors de la phase de génération) et tente de décoder le code-barres.
    Retourne le code complet détecté ou None, avec un message d'erreur en cas d'échec.
    """
    try:
        print(f"   [Décodage] Chargement de l'image : {image_path}")
        image = Image.open(image_path).convert('L')
        img_array = np.array(image)

        # Extraction des points de scan depuis le nom de fichier
        point1, point2 = extract_points_from_filename(os.path.basename(image_path))
        if point1 is None or point2 is None:
            return None, "Points non trouvés dans le nom du fichier"

        x0, y0 = point1
        x1, y1 = point2

        print(f"   Dimensions de l'image : {img_array.shape}")
        print(f"   Points de scan extraits : ({x0}, {y0}) -> ({x1}, {y1})")

        # Extraction de la première signature le long du segment
        num_samples = 2000
        signature, x_coords, y_coords = extract_signature(img_array, x0, y0, x1, y1, num_samples)

        # Calcul de l'histogramme et du seuil d'Otsu (simplifié)
        hist, _ = np.histogram(signature, bins=256, range=(0, 255))
        threshold = otsu_threshold(hist)
        print(f"   Seuil d'Otsu calculé : {threshold}")

        # Binarisation de la première signature
        binary_signature = (signature < threshold).astype(int)

        # Détection des limites utiles de la signature
        left_limit, right_limit = find_limits(binary_signature)
        if left_limit is None or right_limit is None:
            return None, "Limites non trouvées"

        print(f"   Limites trouvées : gauche={left_limit}, droite={right_limit}")

        # Calcul du nombre optimal d'échantillons et du coefficient u
        signature_length = right_limit - left_limit
        num_samples_new, u = calculate_optimal_samples(signature_length)
        print(f"   Coefficient u calculé : {u}")
        print(f"   Nombre d'échantillons optimal : {num_samples_new}")
        
        # Extraction d'une seconde signature le long du rayon utile
        x_left = x_coords[left_limit]
        y_left = y_coords[left_limit]
        x_right = x_coords[right_limit]
        y_right = y_coords[right_limit]
        second_signature, _, _ = extract_signature(img_array, x_left, y_left, x_right, y_right, num_samples_new)
        binary_second_signature = (second_signature < threshold).astype(int)

        # Segmentation et décodage
        segments = segment_barcode(binary_second_signature, u)
        first_digit, first_group_digits = decode_first_group(segments, u)
        second_group_digits = decode_second_group(segments, u)
        
        # Assemblage du code complet
        complete_code = first_digit + ''.join(first_group_digits) + ''.join(second_group_digits)
        print(f"   Code décodé : {complete_code}")
        
        # Création d'une visualisation avec le segment utilisé
        plt.figure(figsize=(12, 8))
        plt.imshow(img_array, cmap='gray')
        plt.plot([x0, x1], [y0, y1], '-', linewidth=1, label='Segment utilisé')
        plt.title(f'Code-barres détecté (exemple) : {complete_code}')
        plt.legend()

        output_path = os.path.join(output_dir, f"decoded_{Path(image_path).stem}_{complete_code}.png")
        plt.savefig(output_path)
        plt.close()

        return complete_code, None
        
    except Exception as e:
        return None, str(e)

def process_directory_decode(input_dir, output_dir):
    """
    Parcourt toutes les images prétraitées dans input_dir, tente de décoder le code-barres et sauvegarde
    les résultats dans output_dir (images annotées et log).
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}

    total_images = 0
    successful_decodes = 0
    failed_decodes = 0
    correct_matches = 0

    with open(log_file, 'w', encoding='utf-8') as log:
        log.write("Traitement des codes-barres - Log\n")
        log.write("=" * 50 + "\n\n")

        for filename in os.listdir(input_dir):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                total_images += 1
                input_path = os.path.join(input_dir, filename)
                
                # Extraction du code attendu (optionnel)
                expected_barcode = extract_barcode_from_filename(filename)
                log.write(f"\nTraitement de: {filename}\n")
                print(f"\n[Décodage] Traitement de {filename}...")
                if expected_barcode:
                    print(f"Code-barres attendu: {expected_barcode}")
                    log.write(f"Code-barres attendu: {expected_barcode}\n")

                detected_barcode, error = process_single_image_decode(input_path, output_dir)

                if detected_barcode:
                    successful_decodes += 1

                    if expected_barcode:
                        if detected_barcode == expected_barcode:
                            correct_matches += 1
                            status = "MATCH"
                        else:
                            status = "MISMATCH"
                        log.write(f"[{status}] Code-barres détecté: {detected_barcode}\n")
                        print(f"[{status}] Code-barres détecté: {detected_barcode}")
                    else:
                        log.write(f"[SUCCESS] Code-barres détecté: {detected_barcode}\n")
                        print(f"[SUCCESS] Code-barres détecté: {detected_barcode}")
                else:
                    failed_decodes += 1
                    log.write(f"[FAILED] Échec de la détection: {error}\n")
                    print(f"[FAILED] Échec de la détection: {error}")
                    
                    # Copie de l'image dans un sous-dossier "failed"
                    output_failed = os.path.join(output_dir, f"failed_{filename}")
                    shutil.copy2(input_path, output_failed)

        log.write("\n" + "=" * 50 + "\n")
        log.write("Statistiques de traitement:\n")
        log.write(f"Total d'images traitées: {total_images}\n")
        log.write(f"Détections réussies: {successful_decodes}\n")
        log.write(f"Détections correctes: {correct_matches}\n")
        log.write(f"Échecs de détection: {failed_decodes}\n")

        if successful_decodes > 0:
            accuracy = (correct_matches / successful_decodes) * 100
            log.write(f"Précision: {accuracy:.2f}%\n")

        print("\n[Décodage] Traitement terminé!")
        print(f"Total d'images traitées: {total_images}")
        print(f"Détections réussies: {successful_decodes}")
        print(f"Détections correctes: {correct_matches}")
        print(f"Échecs de détection: {failed_decodes}")
        if successful_decodes > 0:
            print(f"Précision: {accuracy:.2f}%")
        print(f"Consultez le fichier log pour plus de détails: {log_file}")

def main_decode():
    """
    Fonction 'main' pour le décodage des images “processed_*” (issue de code1_B).
    """
    input_directory = "OUTPUT_BARCODE"
    output_directory = "output_decodage"
    process_directory_decode(input_directory, output_directory)

###############################################################################
#        FONCTION PRINCIPALE : Exécution complète de la chaîne de traitement
###############################################################################
if __name__ == "__main__":
    # 1) Générer les images traitées à partir de "Base_donnes" => "OUTPUT_BARCODE"
    main_generate()

    # 2) Décoder les images dans "OUTPUT_BARCODE" => "output_decodage"
    main_decode()
