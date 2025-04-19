# Requisitos: 
# pip install pillow pytesseract opencv-python

import cv2
import pytesseract
from PIL import Image
import numpy as np

# Configurar caminho do Tesseract (se necessário)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    """Pré-processa a imagem para melhorar o OCR."""
    # Carregar imagem
    img = cv2.imread(image_path)
    
    # Converter para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Redução de ruído
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10)
    
    # Binarização (thresholding adaptativo)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Melhorar contraste (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(thresh)
    
    return enhanced

def extract_text(image_path, lang='por'):
    """Extrai texto de uma imagem com pré-processamento."""
    try:
        # Pré-processamento
        processed_img = preprocess_image(image_path)
        
        # Usar Tesseract com configurações otimizadas
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(
            processed_img,
            lang=lang,
            config=custom_config
        )
        
        return text.strip()
    
    except Exception as e:
        print(f"Erro: {str(e)}")
        return None

# Uso ##################################################
if __name__ == "__main__":
    image_path = "4.png"  # Substituir pelo seu arquivo
    extracted_text = extract_text(image_path)
    
    if extracted_text:
        print("Texto extraído:")
        print("="*50)
        print(extracted_text)
    else:
        print("Falha na extração de texto.")
