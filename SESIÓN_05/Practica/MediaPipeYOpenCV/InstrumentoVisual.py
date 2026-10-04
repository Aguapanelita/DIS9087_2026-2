import cv2
import mediapipe as mp
import numpy as np
import time
import pyaudio
import threading

# --- CONFIGURACIÓN DE PANTALLA ---
ANCHO = 1280
ALTO = 720
ANCHO_PANEL = 280

# --- CONFIGURACIÓN DE AUDIO CON CALLBACK (ESTABLE Y SEGURO) ---
SAMPLE_RATE = 44100
CHANNELS = 1
CHUNK = 1024

frecuencias_activas = []
bloqueo_audio = threading.Lock()
fases_actuales = {}

def audio_callback(in_data, frame_count, time_info, status):
    global frecuencias_activas, fases_actuales
    buffer = np.zeros(frame_count, dtype=np.float32)
    
    with bloqueo_audio:
        notas_locales = list(frecuencias_activas)
        
    if len(notas_locales) > 0:
        num_notas = len(notas_locales)
        t = np.arange(frame_count)
        
        for freq, estilo in notas_locales:
            clave_nota = f"{freq}_{estilo}"
            fase_inicial = fases_actuales.get(clave_nota, 0.0)
            
            omega = 2.0 * np.pi * freq / SAMPLE_RATE
            angulos = fase_inicial + omega * t
            fases_actuales[clave_nota] = (fase_inicial + omega * frame_count) % (2.0 * np.pi)
            
            if estilo == "tecno":
                onda = np.sign(np.sin(angulos)) * 0.12
            elif estilo == "pop":
                onda = np.sin(angulos) * 0.20 + np.sin(angulos * 2) * 0.05
            elif estilo == "rock":
                onda = np.sin(angulos) * 1.2
                onda = np.clip(onda, -0.25, 0.25)
                
            buffer += onda / num_notas
    else:
        fases_actuales.clear()
        
    return (buffer.tobytes(), pyaudio.paContinue)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=CHANNELS, rate=SAMPLE_RATE,
                output=True, frames_per_buffer=CHUNK, stream_callback=audio_callback)

# --- CONFIGURACIÓN DE MEDIAPIPE Y CÁMARA ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
# --- VARIABLES DEL LIENZO ---
lapiz_seleccionado = 0
estilos_musicales = ["tecno", "pop", "rock"]
colores_lapices = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]
codigos_hex = ["#FF6464", "#64FF64", "#6464FF"]

trazos_dibujo = []
ultimo_punto_detectado = None
reproduciendo = False
linea_tiempo_x = ANCHO_PANEL
velocidad_reproduccion = 6
indice_abajo = False  # Indica si el dibujo debe pausarse

def manejar_mouse(event, x, y, flags, param):
    global lapiz_seleccionado, reproduciendo, linea_tiempo_x
    if event == cv2.EVENT_LBUTTONDOWN and x < ANCHO_PANEL:
        for i in range(3):
            y_min = 120 + (i * 140)
            y_max = y_min + 90
            if y_min <= y <= y_max:
                lapiz_seleccionado = i
        if 30 <= x <= 250 and 560 <= y <= 620:
            reproduciendo = not reproduciendo
            if reproduciendo: linea_tiempo_x = ANCHO_PANEL

cv2.namedWindow('Lienzo Musical por Camara')
cv2.setMouseCallback('Lienzo Musical por Camara', manejar_mouse)

while True:
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    
    # 1. EXTRACCIÓN K-MEANS
    zona_muestreo = frame[:, ANCHO_PANEL:]
    img_pequena = cv2.resize(zona_muestreo, (40, 40), interpolation=cv2.INTER_AREA)
    pixeles = img_pequena.reshape((-1, 3))
    pixeles = np.float32(pixeles)
    criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, etiquetas, centros = cv2.kmeans(pixeles, 3, None, criterios, 5, cv2.KMEANS_RANDOM_CENTERS)
    centros = np.uint8(centros)
    _, frecuencias = np.unique(etiquetas, return_counts=True)
    indices_ordenados = np.argsort(frecuencias)[::-1]
    
    for idx, i_ord in enumerate(indices_ordenados):
        b, g, r = centros[i_ord]
        colores_lapices[idx] = (int(b), int(g), int(r))
        codigos_hex[idx] = f"#{r:02x}{g:02x}{b:02x}".upper()

    # 2. RASTREO Y EVALUACIÓN DEL DEDO ÍNDICE
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    dedo_detectado = False
    indice_abajo = False
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark
        
        # ID 8 (Punta del índice) e ID 6 (Nudillo base del índice)
        ix, iy = int(lm[8].x * ANCHO), int(lm[8].y * ALTO)
        
        # EL DEDO ESTÁ ABAJO si la punta (8) está por debajo de su nudillo base (6)
        if lm[8].y > lm[6].y:
            indice_abajo = True

        # Dibuja solo si el índice está arriba y dentro de la zona del lienzo
        if not indice_abajo and ANCHO_PANEL <= ix < ANCHO and 0 <= iy < ALTO:
            dedo_detectado = True
            punto_actual = {'x': ix, 'y': iy, 'lapiz': lapiz_seleccionado}
            if ultimo_punto_detectado is None:
                trazos_dibujo.append([punto_actual])
            else:
                trazos_dibujo[-1].append(punto_actual)
            ultimo_punto_detectado = punto_actual
            
    if not dedo_detectado: 
        ultimo_punto_detectado = None

    # 3. RENDERIZAR TRAZOS
    for trazo in trazos_dibujo:
        if len(trazo) > 1:
            for i in range(1, len(trazo)):
                cv2.line(frame, (trazo[i-1]['x'], trazo[i-1]['y']), (trazo[i]['x'], trazo[i]['y']), colores_lapices[trazo[i]['lapiz']], 6, cv2.LINE_AA)

    # 4. REPRODUCCIÓN E INTERPOLACIÓN
    nuevas_notas = []
    if reproduciendo:
        linea_tiempo_x += velocidad_reproduccion
        if linea_tiempo_x >= ANCHO: linea_tiempo_x = ANCHO_PANEL
        cv2.line(frame, (linea_tiempo_x, 0), (linea_tiempo_x, ALTO), (0, 230, 255), 2)

        for trazo in trazos_dibujo:
            for i in range(1, len(trazo)):
                x1, x2 = trazo[i-1]['x'], trazo[i]['x']
                y1, y2 = trazo[i-1]['y'], trazo[i]['y']
                if min(x1, x2) <= linea_tiempo_x <= max(x1, x2):
                    y_interp = y1 if x2 == x1 else y1 + ((linea_tiempo_x - x1) / (x2 - x1)) * (y2 - y1)
                    porcentaje_altura = 1.0 - (y_interp / ALTO)
                    grados_escala = [130.81, 146.83, 164.81, 196.00, 220.00, 261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33, 659.25, 783.99, 880.00]
                    indice_nota = int(porcentaje_altura * (len(grados_escala) - 1))
                    frecuencia = grados_escala[max(0, min(indice_nota, len(grados_escala)-1))]
                    estilo_actual = estilos_musicales[trazo[i]['lapiz']]
                    if (frecuencia, estilo_actual) not in nuevas_notas:
                        nuevas_notas.append((frecuencia, estilo_actual))

    with bloqueo_audio: frecuencias_activas = nuevas_notas

    # 5. PANEL DE CONTROL (UI)
    cv2.rectangle(frame, (0, 0), (ANCHO_PANEL, ALTO), (25, 25, 25), -1)
    cv2.line(frame, (ANCHO_PANEL, 0), (ANCHO_PANEL, ALTO), (50, 50, 50), 2)
    cv2.putText(frame, "LIENZO MUSICAL", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Estado dinámico basado únicamente en la postura del índice
    if indice_abajo:
        cv2.putText(frame, "[LAPIZ: DETENIDO]", (25, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 255), 2)
    else:
        cv2.putText(frame, "Dibujando...", (25, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
    
    for i in range(3):
        y_inicial = 120 + (i * 140)
        grosor_borde = 3 if i == lapiz_seleccionado else 1
        color_borde = (0, 215, 255) if i == lapiz_seleccionado else (80, 80, 80)
        cv2.rectangle(frame, (30, y_inicial), (100, y_inicial + 70), colores_lapices[i], -1)
        cv2.rectangle(frame, (30, y_inicial), (100, y_inicial + 70), color_borde, grosor_borde)
        cv2.putText(frame, codigos_hex[i], (125, y_inicial + 43), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    color_boton = (40, 180, 40) if not reproduciendo else (40, 40, 180)
    cv2.rectangle(frame, (30, 560), (250, 620), color_boton, -1)
    cv2.putText(frame, "REPRODUCIR" if not reproduciendo else "DETENER", (80, 598), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.imshow('Lienzo Musical por Camara', frame)
    
    tecla = cv2.waitKey(15) & 0xFF
    if tecla == ord('q'): break
    elif tecla == ord('c'): trazos_dibujo.clear(); ultimo_punto_detectado = None
    elif tecla == ord(' '):
        reproduciendo = not reproduciendo
        if reproduciendo: linea_tiempo_x = ANCHO_PANEL

cap.release()
stream.stop_stream(); stream.close(); p.terminate()
cv2.destroyAllWindows()
