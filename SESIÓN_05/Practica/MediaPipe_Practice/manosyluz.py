import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import math

# --- CONFIGURACIÓN DE AUDIO CON EFECTOS Y LUZ ---
SAMP_RATE = 44100
frecuencia_actual = 440.0
volumen_actual = 0.2
nivel_luz = 0.5  # 0.0 (Oscuro/Tenebroso) a 1.0 (Brillante/Tierno)

# Configuración del Delay
DELAY_MAX_SEGUNDOS = 1.0
BUFFER_DELAY_SIZE = int(SAMP_RATE * DELAY_MAX_SEGUNDOS)
buffer_delay = np.zeros(BUFFER_DELAY_SIZE, dtype=np.float32)
ptr_escritura = 0

efecto_mix = 0.0      
efecto_feedback = 0.0 

def audio_callback(outdata, frames, time, status):
    global frecuencia_actual, volumen_actual, efecto_mix, efecto_feedback, ptr_escritura, buffer_delay, nivel_luz
    
    t = (np.arange(frames) + audio_callback.index) / SAMP_RATE
    audio_callback.index += frames
    
    # Modulación de frecuencia por luz (Alta sensibilidad)
    factor_frecuencia = np.interp(nivel_luz, [0.0, 1.0], [0.5, 1.2])
    frec_final = frecuencia_actual * factor_frecuencia
    
    # Onda principal (Limpia)
    onda_pura = np.sin(2 * np.pi * frec_final * t)
    
    # Si está oscuro (nivel_luz bajo), añadimos disonancia tenebrosa (sub-oscilador desafinado)
    if nivel_luz < 0.5:
        factor_tenebroso = np.interp(nivel_luz, [0.0, 0.5], [0.6, 0.0]) 
        # Segunda onda grave y ligeramente desafinada para causar tensión
        onda_tenebrosa = np.sin(2 * np.pi * (frec_final * 0.51) * t) 
        onda_pura = (onda_pura * (1.0 - factor_tenebroso)) + (onda_tenebrosa * factor_tenebroso)
    
    # Aplicar el volumen general
    onda_pura = onda_pura * volumen_actual
    
    # Procesador de Efectos (Delay Line)
    onda_procesada = np.zeros(frames, dtype=np.float32)
    retraso_samples = int(SAMP_RATE * 0.4)
    
    for i in range(frames):
        ptr_lectura = (ptr_escritura - retraso_samples + BUFFER_DELAY_SIZE) % BUFFER_DELAY_SIZE
        sello_eco = buffer_delay[ptr_lectura]
        
        onda_procesada[i] = (1.0 - efecto_mix) * onda_pura[i] + (efecto_mix * sello_eco)
        buffer_delay[ptr_escritura] = onda_pura[i] + (sello_eco * efecto_feedback)
        ptr_escritura = (ptr_escritura + 1) % BUFFER_DELAY_SIZE
        
    outdata[:] = onda_procesada.reshape(-1, 1)

audio_callback.index = 0

stream = sd.OutputStream(channels=1, callback=audio_callback, samplerate=SAMP_RATE)
stream.start()

# --- CONFIGURACIÓN DE MEDIAPIPE HANDS ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calcular_distancia(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

cap = cv2.VideoCapture(0)
print("¡Sintetizador Unificado Iniciado! Presiona 'q' para salir.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
        
    frame = cv2.flip(frame, 1)
    alto, ancho, _ = frame.shape
    
    # --- CÁLCULO DE LUZ EN LA ESCENA (ALTA SENSIBILIDAD) ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    luz_promedio = np.mean(gray) 
    
    # Rango reducido para reaccionar agresivamente a pequeños cambios lumínicos
    target_luz = np.interp(luz_promedio, [40.0, 130.0], [0.0, 1.0]) 
    nivel_luz = nivel_luz * 0.7 + target_luz * 0.3
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    mano_der_detectada = False
    mano_izq_detectada = False
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            etiqueta_mano = handedness.classification[0].label 
            
            puntos = []
            for lm in hand_landmarks.landmark:
                puntos.append((int(lm.x * ancho), int(lm.y * alto)))
            
            p_pulgar = puntos[4]
            p_indice = puntos[8]
            
            # Dibujar marcas de las manos en el fotograma principal
            cv2.circle(frame, p_pulgar, 6, (0, 255, 0), -1)
            cv2.circle(frame, p_indice, 6, (255, 0, 0), -1)
            cv2.line(frame, p_pulgar, p_indice, (255, 255, 255), 2)
            
            dist_px = calcular_distancia(p_pulgar, p_indice)
            
            if etiqueta_mano == "Right":
                mano_der_detectada = True
                frec_target = np.interp(dist_px, [20, 180], [200.0, 800.0])
                frecuencia_actual = frecuencia_actual * 0.7 + frec_target * 0.3
                
            elif etiqueta_mano == "Left":
                mano_izq_detectada = True
                mix_target = np.interp(dist_px, [20, 180], [0.0, 0.85])
                feed_target = np.interp(dist_px, [20, 180], [0.0, 0.80])
                
                efecto_mix = efecto_mix * 0.6 + mix_target * 0.4
                efecto_feedback = efecto_feedback * 0.6 + feed_target * 0.4

    if not mano_der_detectada:
        volumen_actual = volumen_actual * 0.8 + 0.0 * 0.2
    else:
        volumen_actual = volumen_actual * 0.8 + 0.2 * 0.2 
        
    # --- DIBUJAR LECTURAS DIRECTAMENTE EN EL FRAME DE LA CÁMARA ---
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    
    # Fondo semi-transparente para que el texto sea legible sobre la cámara
    cv2.rectangle(frame, (10, 10), (380, 120), (0, 0, 0), -1)
    
    # Línea 1: Frecuencia de la nota
    cv2.putText(frame, f"Nota Base: {int(frecuencia_actual)} Hz", (20, 35), fuente, 0.5, (255, 255, 255), 1)
    cv2.rectangle(frame, (150, 23), (150 + int(np.interp(frecuencia_actual, [200, 800], [0, 200])), 35), (255, 0, 0), -1)
    
    # Línea 2: Espacio / Eco
    cv2.putText(frame, f"Espacio: {int(efecto_mix * 100)}%", (20, 65), fuente, 0.5, (255, 255, 255), 1)
    cv2.rectangle(frame, (150, 53), (150 + int(efecto_mix * 200), 65), (0, 255, 0), -1)
    
    # Línea 3: Cantidad de Luz en la escena
    cv2.putText(frame, f"Luz: {int(nivel_luz * 100)}%", (20, 95), fuente, 0.5, (255, 255, 255), 1)
    cv2.rectangle(frame, (150, 83), (150 + int(nivel_luz * 200), 95), (0, 200, 255), -1)

    # Mostrar únicamente la ventana de la cámara principal
    cv2.imshow('Camara - Control Aereo', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

stream.stop()
stream.close()
cap.release()
cv2.destroyAllWindows()