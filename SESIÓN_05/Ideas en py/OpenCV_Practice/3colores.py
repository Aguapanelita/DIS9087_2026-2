import cv2
import numpy as np

# --- CONFIGURACIÓN DE LA VENTANA ---
ANCHO_PANEL = 500
ALTO_PANEL = 600

def obtener_nombre_color(r, g, b):
    """Convierte un color RGB a HSV para clasificarlo en un espectro amplio."""
    pixel_bgr = np.uint8([[[b, g, r]]])
    pixel_hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)
    
    h = pixel_hsv[0][0][0] * 2  # Escala de 0 a 360°
    s = (pixel_hsv[0][0][1] / 255.0) * 100
    v = (pixel_hsv[0][0][2] / 255.0) * 100

    if v < 15:
        return "negro"
    if v > 85 and s < 10:
        return "blanco"
    if s < 12:
        return "gris"

    if h >= 0 and h < 20:
        if v < 50 and s > 20: return "marron"
        elif v > 70 and s < 40: return "beige"
        return "rojo"
    elif h >= 20 and h < 45:
        if v < 45: return "marron"
        return "naranja"
    elif h >= 45 and h < 70:
        if v > 80 and s < 30: return "crema"
        return "amarillo"
    elif h >= 70 and h < 155:
        if v < 40: return "verde oscuro"
        return "verde"
    elif h >= 155 and h < 185:
        return "cian / turquesa"
    elif h >= 185 and h < 255:
        if v < 40: return "azul marino"
        return "azul"
    elif h >= 255 and h < 290:
        return "morado / violeta"
    elif h >= 290 and h < 335:
        return "magenta / fucsia"
    elif h >= 335 and h <= 360:
        if s < 50 and v > 60: return "rosado"
        return "rojo"

    return "indefinido"

cap = cv2.VideoCapture(0)
print("¡Analizador de Paleta Iniciado! Presiona 'q' para salir.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: 
        break
        
    # --- 1. PROCESAMIENTO K-MEANS PARA EXTRAER LA PALETA PRINCIPAL ---
    img_pequena = cv2.resize(frame, (100, 100), interpolation=cv2.INTER_AREA)
    pixeles = img_pequena.reshape((-1, 3))
    pixeles = np.float32(pixeles)

    criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 3
    _, etiquetas, centros = cv2.kmeans(pixeles, k, None, criterios, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centros = np.uint8(centros)
    
    # Ordenar los colores por la cantidad de píxeles que ocupan (dominancia)
    _, frecuencias = np.unique(etiquetas, return_counts=True)
    indices_ordenados = np.argsort(frecuencias)[::-1]
    centros_ordenados = centros[indices_ordenados]

    # --- 2. CREACIÓN DE LA PANTALLA NEGRA ---
    panel = np.zeros((ALTO_PANEL, ANCHO_PANEL, 3), dtype=np.uint8)
    fuente = cv2.FONT_HERSHEY_SIMPLEX

    # Título principal
    cv2.putText(panel, "PALETA DE COLORES PRINCIPALES", (40, 50), fuente, 0.6, (255, 255, 255), 2)
    
    # --- 3. DIBUJAR LOS 3 COLORES ---
    y_inicial = 120
    espaciado_y = 140

    for i, color_bgr in enumerate(centros_ordenados):
        b, g, r = int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])
        
        codigo_hex = f"#{r:02x}{g:02x}{b:02x}"
        nombre_color = obtener_nombre_color(r, g, b)
        
        y_actual = y_inicial + (i * espaciado_y)
        
        # Dibujar cuadrado de muestra
        cv2.rectangle(panel, (40, y_actual), (140, y_actual + 90), (b, g, r), -1)
        cv2.rectangle(panel, (40, y_actual), (140, y_actual + 90), (255, 255, 255), 1)
        
        # Renderizar textos (Modificado para mostrar únicamente "Color X")
        cv2.putText(panel, f"Color {i+1}", (170, y_actual + 25), fuente, 0.45, (150, 150, 150), 1)
        cv2.putText(panel, nombre_color.upper(), (170, y_actual + 53), fuente, 0.65, (255, 255, 255), 2)
        cv2.putText(panel, codigo_hex, (170, y_actual + 80), fuente, 0.55, (0, 190, 255), 1)

    # --- 4. MOSTRAR INTERFAZ ---
    cv2.imshow('Paleta de Colores K-Means', panel)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
