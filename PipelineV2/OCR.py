import os
import json
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Répertoires de sortie
OUTPUT_DIRECTORY = "../Assets/output"
OCR_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "ocr_data")
PAGES_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "page_images")
TABLES_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "extracted_tables")

os.makedirs(OCR_DIRECTORY, exist_ok=True)
os.makedirs(PAGES_DIRECTORY, exist_ok=True)
os.makedirs(TABLES_DIRECTORY, exist_ok=True)

def detect_tables(image):
    """
    Détecte les tableaux dans une image.
    
    Args:
        image: Image PIL
        
    Returns:
        list: Liste des coordonnées des tableaux détectés [(x, y, w, h), ...]
    """
    # Convertir l'image PIL en format OpenCV
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    
    # Appliquer un flou gaussien pour réduire le bruit
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Détection des bords
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Dilatation pour connecter les lignes de la grille
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Trouver les contours
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrer les contours pour trouver les tableaux potentiels
    tables = []
    for contour in contours:
        # Approximation polygonale
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Si c'est un rectangle et suffisamment grand, c'est potentiellement un tableau
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filtrer selon la taille (éviter les petits rectangles)
            min_size = min(image.size) * 0.1  # Au moins 10% de la dimension minimale de l'image
            if w > min_size and h > min_size:
                tables.append((x, y, w, h))
    
    return tables

def extract_table_text(image, table_coords):
    """
    Extrait le texte d'un tableau détecté.
    
    Args:
        image: Image PIL
        table_coords: Tuple (x, y, w, h) définissant les coordonnées du tableau
        
    Returns:
        str: Texte extrait du tableau
    """
    x, y, w, h = table_coords
    
    # Recadrer l'image au niveau du tableau
    table_image = image.crop((x, y, x+w, y+h))
    
    # OCR spécifique pour les tableaux (configurer tesseract pour les tableaux)
    # --psm 6: Suppose un bloc de texte uniforme
    # --oem 3: Mode de moteur par défaut
    table_text = pytesseract.image_to_string(
        table_image, 
        config='--psm 6 --oem 3'
    )
    
    return table_text

def process_pdf_with_ocr(pdf_path):
    """
    Effectue l'OCR sur un document PDF avec détection de tableaux.
    
    Args:
        pdf_path (str): Chemin vers le fichier PDF
    
    Returns:
        str: Chemin vers le fichier JSON généré
    """
    # Récupérer le nom du document sans extension
    doc_name = Path(pdf_path).stem
    
    # Convertir le PDF en images
    logger.info(f"Conversion du PDF {doc_name} en images...")
    images = convert_from_path(pdf_path)
    logger.info(f"PDF converti en {len(images)} pages.")
    
    # Préparer la structure de sortie
    document_data = {
        "document_name": doc_name,
        "total_pages": len(images),
        "pages": []
    }
    
    # Traiter chaque page
    for i, image in enumerate(images):
        page_num = i + 1
        
        # Sauvegarder l'image de la page
        image_filename = f"{doc_name}_page_{page_num}.jpg"
        image_path = os.path.join(PAGES_DIRECTORY, image_filename)
        image.save(image_path)
        
        # Effectuer l'OCR général
        logger.info(f"Traitement OCR de la page {page_num}...")
        text = pytesseract.image_to_string(image)
        
        # Détecter les tableaux
        logger.info(f"Détection des tableaux sur la page {page_num}...")
        tables = detect_tables(image)
        
        table_data = []
        for j, table_coords in enumerate(tables):
            x, y, w, h = table_coords
            
            # Extraire et enregistrer l'image du tableau
            table_image_filename = f"{doc_name}_page_{page_num}_table_{j+1}.jpg"
            table_image_path = os.path.join(TABLES_DIRECTORY, table_image_filename)
            table_image = image.crop((x, y, x+w, y+h))
            table_image.save(table_image_path)
            
            # Extraire le texte du tableau
            table_text = extract_table_text(image, table_coords)
            
            # Stocker les données du tableau
            table_info = {
                "id": j + 1,
                "coordinates": {"x": x, "y": y, "width": w, "height": h},
                "text": table_text,
                "image_path": table_image_path
            }
            table_data.append(table_info)
            
            logger.info(f"Tableau {j+1} détecté et extrait sur la page {page_num}")
        
        # Ajouter les données de la page au document
        page_data = {
            "page_number": page_num,
            "text": text,
            "image_path": image_path,
            "tables": table_data,
            "tables_count": len(table_data)
        }
        document_data["pages"].append(page_data)
    
    # Enregistrer les données au format JSON
    output_json = os.path.join(OCR_DIRECTORY, f"{doc_name}_ocr_data.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(document_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Données OCR enregistrées dans {output_json}")
    return output_json

def main():
    pdf_path = "/Users/rayanebouaita/Documents/CentraleSupélec/PFE/OpenFinanceAI/Assets/data_test/pdfs/AMEX_EMR_2023.pdf"
    json_path = process_pdf_with_ocr(pdf_path)
    logger.info(f"Traitement OCR terminé. Résultats dans: {json_path}")

if __name__ == "__main__":
    main()