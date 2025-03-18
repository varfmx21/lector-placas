import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import easyocr  # Reemplazando pytesseract con easyocr

# Reemplaza esta ruta con la ubicación de tu imagen
img = cv.imread('image-1.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# 1. Aplicar umbralización binaria con umbral muy alto para capturar solo los pixels muy negros
_, text_mask = cv.threshold(img, 50, 255, cv.THRESH_BINARY_INV)  # Ajustado a 50 para mejor detección

# 2. Aplicar operación morfológica para eliminar ruido pequeño
kernel = np.ones((1, 1), np.uint8)
cleaned_text = cv.morphologyEx(text_mask, cv.MORPH_OPEN, kernel)

# 3. Crear una máscara binaria donde solo está el texto (para visualizar mejor)
text_only = cv.bitwise_and(img, img, mask=cleaned_text)

# 4. Preparar imagen para OCR (formato estándar)
# EasyOCR puede trabajar con ambos formatos, pero generalmente prefiere texto blanco sobre fondo negro
# Mantenemos la imagen binaria con texto en blanco para mejor precisión
ocr_image = cleaned_text

# 5. Aplicar un poco de dilatación para conectar componentes de texto que podrían estar separados
kernel_dilate = np.ones((2, 2), np.uint8)
ocr_image = cv.dilate(ocr_image, kernel_dilate, iterations=1)

# 6. Realizar OCR en la imagen procesada usando EasyOCR
try:
    # Inicializar el lector de EasyOCR para español (puedes cambiarlo a otros idiomas)
    reader = easyocr.Reader(['es', 'en'])  # Admite detección en español e inglés
    
    # Realizar el reconocimiento de texto
    # Convertimos nuestra imagen procesada a formato PIL para EasyOCR
    results = reader.readtext(ocr_image)
    
    # Extraer texto de los resultados
    text_detected = "\n".join([text for _, text, _ in results])
    print("Texto detectado:")
    print(text_detected)
except Exception as e:
    print(f"Error al realizar OCR: {e}")
    print("Asegúrate de tener instalado EasyOCR: pip install easyocr")

# Mostrar resultados
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2), plt.imshow(text_mask, cmap='gray')
plt.title('Texto Extraído (Umbral)'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3), plt.imshow(cleaned_text, cmap='gray')
plt.title('Texto Limpio'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4), plt.imshow(ocr_image, cmap='gray')
plt.title('Imagen Preparada para OCR'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()