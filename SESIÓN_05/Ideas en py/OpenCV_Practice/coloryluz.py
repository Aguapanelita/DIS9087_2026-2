import cv2
import numpy as np

# --- CONFIGURACIÓN DE LA VENTANA ---
ANCHO_PANEL = 640
ALTO_PANEL = 480

def obtener_nombre_color(r, g, b):
    """
    Convierte un color RGB a HSV para clasificarlo en un espectro amplio
    incluyendo marrones, naranjas, rosados, cianes, etc.
    """
    # Creamos una matriz de 1x1 con el color para usar la conversión nativa de OpenCV
    pixel_bgr = np.uint8([[[b, g, r]]])
    pixel_hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)[0][0]
    
    h = pixel_hsv[0] * 2  # OpenCV escala el tono (H) de 0-179, lo multiplicamos por 2 para tener 0-360°
    s = (pixel_hsv[1] / 255.0) * 100 # Saturación en %
    v = (pixel_hsv[2] / 255.0) * 100 # Brillo/Valor en %

    # 1. Casos extremos: Sin saturación o muy oscuros/claros
    if v < 12:
        return "negro"
    if v > 85 and s < 10:
        return "blanco"
    if s < 12:
        return "gris"

    # 2. Espectro amplio basado en el Tono (Hue) y matices especiales
    if h >= 0 and h < 20:
        # El marrón y el beige son tonos naranjas/rojos pero oscuros o apagados
        if v < 50 and s > 20:
            return "marron"
        elif v > 70 and s < 40:
            return "beige"
        return "rojo"
        
    elif h >= 20 and h < 45:
        if v < 45:
            return "marron"
        return "naranja"
        
    elif h >= 45 and h < 70:
        if v > 80 and s < 30:
            return "crema"
        return "amarillo"
        
    elif h >= 70 and h < 155:
        if v < 40:
            return "verde oscuro"
        return "verde"
        
    elif h >= 155 and h < 185:
        return "cian / turquesa"
        
    elif h >= 185 and h < 255:
        if v < 40:
            return "azul marino"
        return "azul"
        
    elif h >= 255 and h < 290:
        return "morado / violeta"
        
    elif h >= 290 and h < 335:
        return "magenta / fucsia"
        
    elif h >= 335 and h <= 360:
        if s < 50 and v > 60:
            return "rosado"
        return "rojo"

    return "indefinido"

cap = cv2.VideoCapture(0)
print("¡Analizador de Espectro Amplio Iniciado! Presiona 'q' para salir.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: 
        break
        
    # --- 1. CÁLCULO DEL PORCENTAJE DE LUMINOSIDAD ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    luz_promedio = np.mean(gray) 
    porcentaje_luminosidad = (luz_promedio / 255.0) * 100

    # --- 2. CÁLCULO DEL COLOR PROMEDIO GLOBAL ---
    pixel_promedio = cv2.resize(frame, (1, 1), interpolation=cv2.INTER_AREA)
    b_val, g_val, r_val = pixel_promedio[0][0]
    
    # Asegurar valores enteros nativos de Python para evitar errores visuales
    r, g, b = int(r_val), int(g_val), int(b_val)
    
    # --- 3. CONVERSIÓN A CÓDIGO HEXADECIMAL Y NOMBRE AMPLIO ---
    codigo_hex = f"#{r:02x}{g:02x}{b:02x}"
    nombre_color = obtener_nombre_color(r, g, b)

    # --- 4. CREACIÓN DE LA PANTALLA NEGRA ---
    panel = np.zeros((ALTO_PANEL, ANCHO_PANEL, 3), dtype=np.uint8)
    fuente = cv2.FONT_HERSHEY_SIMPLEX

    # Título principal
    cv2.putText(panel, "TELEMETRIA DE COLOR Y LUZ EXTENDIDA", (50, 50), fuente, 0.6, (255, 255, 255), 2)
    
    # Renderizar Datos de Luminosidad
    cv2.putText(panel, f"Luminosidad: {porcentaje_luminosidad:.1f}%", (50, 110), fuente, 0.55, (255, 255, 255), 1)
    cv2.rectangle(panel, (50, 125), (450, 135), (40, 40, 40), -1) 
    cv2.rectangle(panel, (50, 125), (50 + int(porcentaje_luminosidad * 4), 135), (255, 145, 0), -1) 

    # --- DISEÑO DEL CÍRCULO CENTRAL (ESTILO TU EJEMPLO) ---
    centro_x, centro_y = ANCHO_PANEL // 2, 275
    radio = 90
    
    # Círculo relleno con el color exacto absorbido
    cv2.circle(panel, (centro_x, centro_y), radio, (b, g, r), -1)
    # Brillo exterior sutil
    cv2.circle(panel, (centro_x, centro_y), radio + 1, (200, 200, 200), 1)

    # Texto con formato exacto: "nombre - #hex"
    texto_final = f"{nombre_color} - {codigo_hex}"
    
    # Calcular ancho exacto del texto para que quede centrado simétricamente
    (t_ancho, t_alto), _ = cv2.getTextSize(texto_final, fuente, 0.6, 1)
    pos_x = centro_x - (t_ancho // 2)
    
    # Imprimir el texto debajo del círculo
    cv2.putText(panel, texto_final, (pos_x, centro_y + radio + 40), fuente, 0.6, (255, 255, 255), 1)

    # --- 5. MOSTRAR INTERFAZ ---
    cv2.imshow('Analisis de Escena HEX', panel)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
